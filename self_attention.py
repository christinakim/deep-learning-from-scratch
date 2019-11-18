import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm as tqdm

import torch
import numpy as np


## inputs key, query, abnd value vector which are the encoder inputs * k, q, v weights from prev training (MLP for key, mLP for value, MLP for query)


def self_attention(key, query, value):
    # scores = dotprod of key + query
    # b here stands for batch, l = length, k= num pixels, num words etc)
    scores = torch.einsum("bkl,bql->bqk", key, query)

    # note that the dimensions of key query and value
    dimensions_k = key.dim()

    # this helps to create a more stable gradient (you could probably normalize via other constants?)
    scores = scores / np.sqrt(dimensions_k)

    # turn scores into probabilities from 0,1
    attention = torch.softmax(scores)

    # dotprod of value vector, and attention vector
    attended_values = torch.einsum("bdl,bad->bal", value, attention)

    return attended_values


"""Testing"""

batch_size = 1
d_model = 3
length = 4

def example_tensor(shape):
  rng = torch.arange(start=0, end=np.product(shape), dtype=torch.float32)
  return torch.reshape(rng, shape)


value = example_tensor([batch_size, d_model, length])
print(f"value: \n{value}\n")

key = torch.zeros([batch_size, d_model, length])
key[0,0,0] = 100.
key[0,1,1] = 100.
key[0,2,2] = 100.
print(f"key: \n{key}\n")


query = torch.zeros([batch_size, d_model, length])
query[0,0,1] = 100.
query[0,1,0] = 100.
query[0,2,2] = 100.
print(f"query: \n{query}\n")

import numpy.testing as npt

attended_values = self_attention(query, key, value)
print(attended_values)

npt.assert_equal(attended_values.numpy(), np.array(
[[[4.,  5.,  6.,  7.],
  [0.,  1.,  2.,  3.],
 [8.,  9., 10., 11.]]]))
