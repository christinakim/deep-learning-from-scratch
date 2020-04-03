import torch
from torch import nn


class GameStateTensor(torch.Tensor):
    pass


class LSTM(nn.Module):
    """
    The three dimensions of the input to this model are:
    Samples. One sequence is one sample. A batch is comprised of one or more samples.
    Time Steps. One time step is one point of observation in the sample.
    Features. One feature is one observation at a time step.

    This LSTM is learning to predict the next gamestate - originally written for a small textworld experiment I was running
    """

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
        )

    def forward(self, input: GameStateTensor):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input)
        final_pred = torch.narrow(lstm_out, 1, 4, 1)
        # Only take the output from the final timetep
        # this should take in the last prediction

        y_pred = self.linear(final_pred)
        # assert size is correct

        return y_pred
