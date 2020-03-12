import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch import tensor
from torch.nn import functional as F

from models.train_utils import debug_grad


class TypeASquareMaskedLinear(nn.Linear):
    def __init__(self, features: int, cardinality, dimensionality, device):
        super().__init__(features, features)
        self.device = device
        self.cardinality, self.dimensionality = cardinality, dimensionality
        masked_area = torch.tril(torch.ones((self.dimensionality, self.dimensionality)), diagonal=-1)
        intermediate_mask = torch.repeat_interleave(masked_area, self.cardinality, dim=0)
        self.mask = nn.Parameter(torch.repeat_interleave(intermediate_mask, self.cardinality, dim=1), requires_grad=False)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs.to(self.device)
        # TODO: Check to see if we need to multiply by weight instead.
        return F.linear(inputs, self.weight * self.mask, self.bias)


class TypeBSquareMaskedLinear(nn.Linear):
    def __init__(self, features: int, cardinality, dimensionality, device):
        super().__init__(features, features)
        self.device = device
        self.cardinality, self.dimensionality = cardinality, dimensionality
        masked_area = torch.tril(torch.ones((self.dimensionality, self.dimensionality)), diagonal=0)
        intermediate_mask = torch.repeat_interleave(masked_area, self.cardinality, dim=0)
        self.register_buffer("mask", torch.repeat_interleave(intermediate_mask, self.cardinality, dim=1))

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs.to(self.device)
        # TODO: Check to see if we need to multiply by weight instead.
        return F.linear(inputs, self.weight * self.mask, self.bias)


class OneHotEncodingLayer(nn.Module):
    def __init__(self, size, cardinality):
        super().__init__()
        self.size = size
        self.cardinality = cardinality
        self.lookup = nn.Parameter(torch.eye(cardinality), requires_grad=False)

    def forward(self, batch):
        """
        Takes list of coordinates:
        [
            [x0, x1],
            [x0, x1],
            ...
            [x0, x1] # batch dimensionality
        ]
        =>
        [
            [x00, x01, x10, x11],
            ...
        ]
        """
        batch = batch.cuda()
        return torch.index_select(self.lookup, 0, batch.flatten()).view(len(batch), self.size * self.cardinality)


class SimpleMADE(nn.Module):
    #: 0, ... d from the problem
    cardinality: int
    #: x0, x1, ..., x_dimensionality
    dimensionality: int

    def __init__(self, d: int):
        super().__init__()
        self.cardinality = d
        self.dimensionality = 2
        self.layers = nn.Sequential(
            OneHotEncodingLayer(self.dimensionality, self.cardinality),
            # Maybe want to change to dimensionality...
            TypeASquareMaskedLinear(self.dimensionality * self.cardinality, self.cardinality, self.dimensionality),
            nn.ReLU(),
            TypeBSquareMaskedLinear(self.dimensionality * self.cardinality, self.cardinality, self.dimensionality),
            nn.ReLU(),
            TypeBSquareMaskedLinear(self.dimensionality * self.cardinality, self.cardinality, self.dimensionality),
            nn.ReLU(),
            TypeBSquareMaskedLinear(self.dimensionality * self.cardinality, self.cardinality, self.dimensionality),
            nn.ReLU(),
            TypeBSquareMaskedLinear(self.dimensionality * self.cardinality, self.cardinality, self.dimensionality),
        )

    def forward(self, batch: Tensor) -> Tensor:
        return F.log_softmax(self.layers(batch).view(len(batch), self.dimensionality, self.cardinality), dim=-1)

    def loss(self, outputs, batch):
        return F.nll_loss(outputs.permute(0, 2, 1), batch)

    def distribution(self) -> np.array:
        with torch.no_grad():
            inputs = tensor([[x_0, 0] for x_0 in range(0, self.cardinality)])
            outputs = self.forward(inputs)
            p_x0 = torch.exp(outputs[0, 0, :])
            p_x1s = torch.exp(outputs[:, 1, :])
            return (p_x0.unsqueeze(-1) * p_x1s).cpu().numpy()


class SimpleMNISTMADE(nn.Module):
    #: 0, ... d from the problem
    cardinality: int
    #: x0, x1, ..., x_dimensionality
    dimensionality: int

    def __init__(self, d: int, h, w, device):
        super().__init__()
        self.cardinality = 1
        self.dimensionality = h * w
        self.h = h
        self.w = w
        self.device = device
        self.layers = nn.Sequential(
            TypeASquareMaskedLinear(
                self.dimensionality * self.cardinality, self.cardinality, self.dimensionality, device
            ),
            nn.ReLU(),
            TypeBSquareMaskedLinear(
                self.dimensionality * self.cardinality, self.cardinality, self.dimensionality, device
            ),
            nn.ReLU(),
            TypeBSquareMaskedLinear(
                self.dimensionality * self.cardinality, self.cardinality, self.dimensionality, device
            ),
        )
        self.layers.register_backward_hook(debug_grad)

    def forward(self, batch: Tensor) -> Tensor:
        batch = batch.view(batch.shape[0], self.dimensionality)
        return F.logsigmoid(self.layers(batch).view(len(batch), self.dimensionality))

    def loss(self, outputs, batch):
        return F.binary_cross_entropy(outputs, batch.view(batch.shape[0], self.dimensionality))

    def distribution(self) -> np.array:
        with torch.no_grad():
            inputs = tensor([[x_0, 0] for x_0 in range(0, self.dimensionality)]).to(self.device)
            outputs = self.forward(inputs)
            p_x0 = torch.exp(outputs[0, 0, :])
            p_x1s = torch.exp(outputs[:, 1, :])
            return (p_x0.unsqueeze(-1) * p_x1s).cpu().numpy()

    def sample(self, num_samples) -> tensor:
        with torch.no_grad():
            empty = torch.zeros((num_samples, self.h * self.w))
            for i in range(self.dimensionality):
                output = self(empty.view(num_samples, self.h, self.w))
                empty[:, i] = torch.bernoulli(torch.exp(output[:, i]))
        return empty.view(num_samples, self.h, self.w, 1)

    def test_masks(self):
        mask_products = torch.ones((self.dimensionality, self.dimensionality)).to(self.device)
        for layer in reversed(list(self.layers.children())):
            if isinstance(layer, (TypeASquareMaskedLinear, TypeBSquareMaskedLinear)):
                mask_products *= layer.mask
        return mask_products
