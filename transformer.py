import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm as tqdm

# default hyperparameters for training, with 0.1 learning rate, 0.1 dropout, and early stopping
from mlp import Conv1d
from mlp import MLP

batch_size = 512
epochs = 10
lr = 0.001
dropout = 0.001
early_stopping = True
seed = 42
log_interval = 10
no_cuda = False
attentional_pooling = False


# Model params
n_layers = 3
n_bins = 8
n_states = 256
n_classes = 10
n_pixels = 14 * 14  # 2x2 subsampled MNIST
n_embed_vals = 16 * 16
n_heads = 4
assert (
    n_states % n_heads == 0
), f"Must divide {n_states} states among the {n_heads} heads"


class ChannelNorm(nn.Module):
    """Normalize each channel with layernorm"""

    def __init__(self):
        super(ChannelNorm, self).__init__()
        # Normalize all elements of each channel
        # TODO: Try reversing this
        self.layer_norm = nn.LayerNorm(normalized_shape=[n_states])

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return x.transpose(2, 1)


def dotprod(a, b):
    return a.matmul(b.transpose(-1, -2))


class MultiheadAttention(nn.Module):
    def __init__(self):
        super(MultiheadAttention, self).__init__()

        self.fc_q = Conv1d(in_channels=n_states, out_channels=n_states)
        self.fc_k = Conv1d(in_channels=n_states, out_channels=n_states)
        self.fc_v = Conv1d(in_channels=n_states, out_channels=n_states)

        self.fc_out = Conv1d(in_channels=n_states, out_channels=n_states)

    def forward(self, x):
        def _split_heads_and_transpose_to_states_last(x):
            """(batch, state, pixel) => (batch, head, pixel, head_state)"""
            x = x.view([batch_size, n_heads, n_states // n_heads, n_pixels])
            return x.transpose(-1, -2)

        def _merge_heads_and_transpose_to_pixels_last(x):
            """(batch, head, pixel, head_state) => (batch, state, pixel)"""
            x = x.view([batch_size, n_pixels, n_states])
            return x.transpose(-1, -2)

        query = _split_heads_and_transpose_to_states_last(self.fc_q(x))
        keys = _split_heads_and_transpose_to_states_last(self.fc_k(x))
        vals = _split_heads_and_transpose_to_states_last(self.fc_v(x))

        # How much is each pixel attending to each other pixel?
        att_w = dotprod(query, keys)
        assert att_w.shape == (batch_size, n_heads, n_pixels, n_pixels)

        att_w = att_w / np.sqrt(n_states / n_heads)  # Rescale by head dimension
        att_w = nn.Softmax(dim=-1)(
            att_w
        )  # Normalize attention distribution over the pixels

        # Apply attention weights to values
        att_vals = dotprod(att_w, vals.transpose(-1, -2))  # Reweighted attention value
        att_vals = _merge_heads_and_transpose_to_pixels_last(
            att_vals
        )  # Combine the results

        assert att_vals.shape == (batch_size, n_states, n_pixels)
        return self.fc_out(att_vals)


class TransformerBlock(nn.Module):
    def __init__(self):
        super(TransformerBlock, self).__init__()

        self.attention = MultiheadAttention()
        self.channel_norm1 = ChannelNorm()
        self.mlp = MLP()
        self.channel_norm2 = ChannelNorm()

    def forward(self, x):
        a = self.attention(x)
        a = self.channel_norm1(a + x)
        m = self.mlp(a)
        return self.channel_norm2(m + x)


class EmbedWithPositionalBias(nn.Module):
    """
  Embed discrete inputs to a continuous space and add learned position embeddings.
  The learned position embeddings are crucial for letting the subsequent MLPs access the model
  """

    def __init__(self):
        super(EmbedWithPositionalBias, self).__init__()

        self.x_embed = nn.Embedding(n_embed_vals, embedding_dim=n_states)
        self.x_embed.weight.data.normal_(0.0, 0.02)

        self.pos_embed = nn.Parameter(
            torch.zeros(n_pixels, n_states).normal_(0.0, 0.01)
        )

    def forward(self, x):
        if attentional_pooling:
            # Include an input with a special value (the other pixels will have a value less than n_bins)
            classifier_input = torch.ones(batch_size, 1, dtype=torch.long) * n_bins
            x = torch.cat((x, classifier_input.cuda()), dim=1)

        embedded = self.x_embed(x) + self.pos_embed
        return embedded.transpose(1, 2)  # Return in NCD (batch_size, n_state, n_x)


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.embed = EmbedWithPositionalBias()
        self.transformer_blocks = nn.Sequential(
            *([TransformerBlock()] * n_layers)
        )  # Stack the block n_layers times
        self.fc_final = nn.Linear(n_states, n_classes)

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer_blocks(x)

        # Map from [batch, state, pixel] => [batch, state, logit]
        if attentional_pooling:
            x = x[:, :, 0]  # Just slice out our extra classifier values
        else:
            x = x.mean(dim=2)  # Take the average pool along the pixels dimensions,

        x = self.fc_final(x)  # Map [batch, state] => [batch, logit]
        return F.log_softmax(x, dim=1)  # And do a softmax


def train(model, device, train_loader, optimizer, epoch):
    model.train()

    losses = collections.deque([], 20)  # Smooth out the loss for reporting
    accs = collections.deque([], 20)
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (x, y) in progress_bar:
        x, y = x.to(device).float(), y.to(device)
        optimizer.zero_grad()
        logits = model(x)

        loss = F.cross_entropy(logits, y)
        loss.backward()

        pred = torch.argmax(logits, 1)
        correct = (pred == y).float()
        accs.append((correct.mean().cpu().numpy()) * 100)
        losses.append(loss.data.cpu().numpy())

        progress_bar.set_postfix(
            {
                "epoch": epoch,
                "loss": f"{np.mean(losses):.3f}",
                "accuracy": f"{np.mean(accs) : .2f}%",
            }
        )
        optimizer.step()
