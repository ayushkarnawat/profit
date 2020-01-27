"""
[WIP]: Need to implement both convolutions for vectors and scalars, a 
graph embed layer, gather layer, and nodes layer.
[WIP]: How to incorporate bayesian optimization/combinatorial search 
within the graphical network??? NOTE: Each call to bayesian opt is 
expensive (since it is "online" learning).
"""

from torch import nn, Tensor
from torch.nn import init


class GraphConvS(nn.Module):
    """Graph convolution layer for scalar features.
    
    Each scalar feature :math: `s \\in \\mathbb{R}^M` describes the 
    individual features for each atom in the molecule. 
    """

    def __init__(self):
        in_feats = 10
        out_feats = 1
        self.weight = nn.Parameter(Tensor(in_feats, out_feats))

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)