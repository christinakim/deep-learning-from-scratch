from torch import nn
import torch.nn.functional as F


def Conv1d(in_channels, out_channels):
    """Standard 1D convolution with bias"""
    conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
    # TODO: Remove these initializations and inline the function if it doesn't affect perf
    conv.weight.data.normal_(0.0, 0.02)
    conv.bias.data.zero_()
    return conv


class SidewaysMLP(nn.Module):
    """Two layer relu residual MLP with a wider hidden layer"""

    def __init__(self, n_states: int, ratio: int = 4):
        super(SidewaysMLP, self).__init__()

        self.fc1 = Conv1d(in_channels=n_states, out_channels=ratio * n_states)
        self.fc2 = Conv1d(in_channels=ratio * n_states, out_channels=n_states)

    def forward(self, x):
        hidden = F.relu(self.fc1(x))
        residual = self.fc2(hidden)
        return residual + x


class MLP(nn.Module):
    def __init__(self, n_states: int, n_classes: int, n_hidden: int = 64):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        hidden = F.relu(self.fc1(x))
        hidden = F.relu(self.fc2(hidden))
        logits = F.relu(self.fc3(hidden))

        return logits
