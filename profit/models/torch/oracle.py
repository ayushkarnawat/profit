import torch
from torch import nn
from torch.nn import functional as F


class SequenceOracle(nn.Module):
    """Regular densely connected neural network.

    Params:
    -------
    seqlen: int
        Length of sequence.

    vocab_size: int
        The size of the dictionary, aka number of unique tokens.

    hidden_size: int, default=1024
        Number of features.

    out_size: int, default=2
        Number of outputs, i.e. values to predict.
    """

    def __init__(self,
                 seqlen: int,
                 vocab_size: int,
                 hidden_size: int = 50,
                 out_size: int = 2) -> None:
        super(SequenceOracle, self).__init__()
        self.fc1 = nn.Linear(seqlen * vocab_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        h1 = F.elu(self.fc1(x))
        return self.fc2(h1)
