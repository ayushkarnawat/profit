"""
[WIP]: Need to implement both convolutions for vectors and scalars, a 
graph embed layer, gather layer, and nodes layer.
[WIP]: How to incorporate bayesian optimization/combinatorial search 
within the graphical network??? NOTE: Each call to bayesian opt is 
expensive (since it is "online" learning).
"""

import inspect
from typing import Optional, TypeVar, Union

from torch import nn, Tensor
from torch.nn import init
from torch.nn.modules import activation

activations = [obj for _, obj in inspect.getmembers(activation, inspect.isclass) 
               if obj.__module__ == "torch.nn.modules.activation"]
Activations = TypeVar("activations", *activations) # for type-checking purposes


class GraphConvS(nn.Module):
    """Graph convolution layer for scalar features.
    
    Each scalar feature :math: `s \\in \\mathbb{R}^M` describes the 
    individual features for each atom in the molecule.

    The model params are initialized as in the original implementation, 
    where the weight :math:`W` is initialized using Glorot uniform 
    initialization and the bias is initialized to be zero.

    Attributes
    ----------
    weight : torch.Tensor
        The learnable weight tensor.
    bias : torch.Tensor
        The learnable bias tensor.
    """

    def __init__(self, n_filters: int, bias: bool=True, activation: Optional[Activations]=None):
        super(GraphConvS, self).__init__()

        # Loss is denoted by the kernel_regulizer, which gets activated onto a weight parameter along with the initial weights. 
        assert n_filters > 0, f"Number of filters (N={n_filters}) must be greater than 0."
        self._n_filters = n_filters

        # Add learnable weight/bias params to the layer.
        # NOTE: The in_feats, out_feats should be computed via the shapes of the previous layers.
        # TODO: Dummy values, remove
        in_feats = 10
        self.weight = nn.Parameter(Tensor(in_feats, n_filters), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(Tensor(n_filters))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters() # init weights for 1st optim step

        self._activation = activation


    def reset_parameters(self):
        """Reinitialize all learnable parameters."""
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)


    def forward(self):
        """Compute graph convolution.
        
        Convolve two atom scalar features :math:`s_1, s_2`. Specifically, 
        we (a) concatenate both scalars, (b) convolve features, (c) 
        multiply using the adjacency matrix, (d) downsample through 
        pooling, and (e) apply a non-linear activation.
        """
        pass

    
    def extra_repr(self):
        """String representation of the module. 
        
        Sets the extra representation, which comes into effect when 
        printing the (full) model summary.
        """
        summary = 'n_filters={_n_filters}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)