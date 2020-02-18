"""
[WIP]: Need to implement both convolutions for vectors and scalars, a 
graph embed layer, gather layer, and nodes layer.
[WIP]: How to incorporate bayesian optimization/combinatorial search 
within the graphical network??? NOTE: Each call to bayesian opt is 
expensive (since it is "online" learning).
"""

import inspect
from typing import List, Optional, Tuple, TypeVar, Union

import torch
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

    TODO: How do we handle L1/L2 regularization on the weight kernel? 
    In keras, this was done when the inital kernel weights were 
    specified (see kernel_regularizer parameter within `add_weight`). 

    Params:
    -------
    n_filters: int
        The number of filters/kernels to use. This is the number of 
        atom-level features in a mol.
    
    pooling: str, optional, default="sum"
        Type of down-sample pooling technique to perform on the hidden 
        representations after the graph convolutions. Depending on the 
        pooling type chosen, the representations are etiher summed, 
        averaged, or maxed.
    
    bias: bool, default="sum"
        If True, adds a learnable bias to the convolved output.
    
    activation: callable activation function or None, optional, default=None
        If not None, applies an activation function to the updated node 
        features. See ::module:`torch.nn.modules.activation` for list 
        of all possible activation functions.
    
    Attributes
    ----------
    weight : torch.Tensor
        The learnable weight tensor.
    
    bias : torch.Tensor
        The learnable bias tensor.
    """

    def __init__(self, 
                 n_filters: int, 
                 pooling: str="sum", 
                 bias: bool=True, 
                 activation: Optional[Activations]=None):
        super(GraphConvS, self).__init__()

        assert n_filters > 0, f"Number of filters (N={n_filters}) must be greater than 0."
        self._n_filters = n_filters
        self._pooling = pooling
        if activation is not None:
            assert isinstance(activation, tuple(activations)), f"Invalid " 
            f"activation func type: {type(activation)}. Should be one of the "
            f"following: {activations}."
        self._activation = activation

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

    def reset_parameters(self):
        """Reinitialize all learnable parameters."""
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, inputs: Union[Tensor, List[Tensor], Tuple[Tensor, ...]]) \
                -> torch.Tensor:
        """Compute graph convolution.
        
        Convolve two atom scalar features :math:`s_1, s_2`. Specifically, 
        we (a) concatenate both scalars, (b) convolve features, (c) 
        multiply using the adjacency matrix, (d) downsample through 
        pooling, and (e) apply a non-linear activation.

        Params:
        -------
        inputs: torch.Tensor or list/tuple of torch.Tensors.
            Inputs to the current layer.

        Returns:
        --------
        sfeats: torch.Tensor
            Output tensor after convolution, pooling, and activation.

        Notes:
        ------
        Input shape(s):
        * sfeats1: :math:`(N, M, M, nfeats)` where :math:`N` is the 
            number of samples, :math:`M` is the max number of atoms, 
            and :math:`nfeats` is number of scalar feats in the previous layer.
        * sfeats2: :math:`(N, M, M, nfeats)` where :math:`N` is the 
            number of samples, :math:`M` is the max number of atoms, 
            and :math:`nfeats` is number of scalar feats in the previous layer.
        * adjacency: :math:`(N, M, M)` where :math:`N` is the number of 
            samples, and :math:`M` is the max number of atoms.
        
        Output shape:
        * sfeats: :math:`(N, M, \\text{n_filters})` where :math:`N` is 
            the number of samples, :math:`M` is the max number of atoms.
        """
        # Import graph tensors
        # scalar_feats_1 = (samples, max_atoms, max_atoms, num_atom_feats)
        # scalar_feats_2 = (samples, max_atoms, max_atoms, num_atom_feats)
        # adjacency = (samples, max_atoms, max_atoms)
        scalar_feats_1, scalar_feats_2, adjacency = inputs

        # Get parameters
        # NOTE: This is assuming that the # samples are in channels_first. 
        # Change to support channels_last? 
        N = int(scalar_feats_1.shape[1])    # max number of atoms
        M1 = int(scalar_feats_1.shape[-1])  # num of features in s_1
        M2 = int(scalar_feats_2.shape[-1])  # num of features in s_2

        # 1. Concatenate two features, 4D tensor
        sfeats = torch.cat((scalar_feats_1, scalar_feats_2), dim=-1)

        # 2. Linear combination (aka convolve), 4D tensor
        sfeats = torch.reshape(sfeats, shape=(-1, M1 + M2))
        sfeats = torch.matmul(sfeats, self.weight)
        if self.bias is not None:
            sfeats = sfeats + self.bias
        sfeats = torch.reshape(sfeats, shape=(-1, N, N, self.filters))

        # 3. Adjacency masking, 4D tensor
        adjacency = torch.reshape(adjacency, shape=(-1, N, N, 1))
        adjacency = adjacency.repeat(1, 1, 1, self.filters)
        sfeats = sfeats * adjacency # element-wise multiplication

        # 4. Integrate over second atom axis, 3D tensor
        if self._pooling == "sum":
            sfeats = torch.sum(sfeats, dim=2)
        elif self._pooling == "mean":
            sfeats = torch.mean(sfeats, dim=2)
        elif self._pooling == "max":
            sfeats = torch.max(sfeats, dim=2)

        # 5. Activation, 3D tensor
        if self._activation is not None:
            sfeats = self._activation(sfeats)
        
        return sfeats

    def extra_repr(self) -> str:
        """String representation of the module. 
        
        Sets the extra representation, which comes into effect when 
        printing the (full) model summary.
        """
        summary = 'n_filters={_n_filters}, pooling={_pooling}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)


class GraphConvV(nn.Module):
    """Graph convolution layer for vector features.
    
    Each vector feature :math: `V \\in \\mathbb{R}^{M \\times 3}` 
    describes the vector features (which include 3D for XYZ coords) for 
    each atom in the molecule.

    The model params are initialized as in the original implementation, 
    where the weight :math:`W` is initialized using Glorot uniform 
    initialization and the bias is initialized to be zero.

    TODO: How do we handle L1/L2 regularization on the weight kernel? 
    In keras, this was done when the inital kernel weights were 
    specified (see kernel_regularizer parameter within `add_weight`).

    Params:
    -------
    n_filters: int
        The number of filters/kernels to use.
    
    pooling: str, optional, default="sum"
        Type of down-sample pooling technique to perform on the hidden 
        representations after the graph convolutions. Depending on the 
        pooling type chosen, the representations are etiher summed, 
        averaged, or maxed.
    
    bias: bool, default="sum"
        If True, adds a learnable bias to the convolved output.
    
    activation: callable activation function or None, optional, default=None
        If not None, applies an activation function to the updated node 
        features. See ::module:`torch.nn.modules.activation` for list 
        of all possible activation functions.
    
    Attributes
    ----------
    weight : torch.Tensor
        The learnable weight tensor.
    
    bias : torch.Tensor
        The learnable bias tensor.
    """

    def __init__(self, 
                 n_filters: int, 
                 pooling: str="sum", 
                 bias: bool=True, 
                 activation: Optional[Activations]=None):
        super(GraphConvV, self).__init__()

        assert n_filters > 0, f"Number of filters (N={n_filters}) must be greater than 0."
        self._n_filters = n_filters
        self._pooling = pooling
        if activation is not None:
            assert isinstance(activation, tuple(activations)), f"Invalid " 
            f"activation func type: {type(activation)}. Should be one of the "
            f"following: {activations}."
        self._activation = activation

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

    def reset_parameters(self):
        """Reinitialize all learnable parameters."""
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, inputs: Union[Tensor, List[Tensor], Tuple[Tensor, ...]]) \
                -> torch.Tensor:
        """Compute graph convolution.
        
        Convolve two atom scalar features :math:`V_1, V_2`. Specifically, 
        we (a) concatenate both vectors, (b) convolve features, (c) 
        multiply using the adjacency matrix, (d) downsample through 
        pooling, and (e) apply a non-linear activation.

        Params:
        -------
        inputs: torch.Tensor or list/tuple of torch.Tensors.
            Inputs to the current layer.

        Returns:
        --------
        vfeats: torch.Tensor
            Output tensor after convolution, pooling, and activation.

        Notes:
        ------
        Input shape(s):
        * vfeats1: :math:`(N, M, M, 3, nfeats)` where :math:`N` is the 
            number of samples, :math:`M` is the max number of atoms, 
            and :math:`nfeats` is number of vector feats in the previous layer.
        * vfeats2: :math:`(N, M, M, 3, nfeats)` where :math:`N` is the 
            number of samples, :math:`M` is the max number of atoms, 
            and :math:`nfeats` is number of vector feats in the previous layer.
        * adjacency: :math:`(N, M, M)` where :math:`N` is the number of 
            samples, and :math:`M` is the max number of atoms.
        
        Output shape:
        * vfeats: :math:`(N, M, 3, \\text{n_filters})` where :math:`N` is 
            the number of samples, :math:`M` is the max number of atoms.
        """
        # Import graph tensors
        # vector_feats_1 = (samples, max_atoms, max_atoms, coor_dims, atom_feat)
        # vector_feats_2 = (samples, max_atoms, max_atoms, coor_dims, atom_feat)
        # adjacency = (samples, max_atoms, max_atoms)
        vector_feats_1, vector_feats_2, adjacency = inputs

        # Get parameters
        # NOTE: This is assuming that the # samples are in channels_first. 
        # Change to support channels_last? 
        N = int(vector_feats_1.shape[1])    # max number of atoms
        M1 = int(vector_feats_1.shape[-1])  # num of features in V_1
        M2 = int(vector_feats_2.shape[-1])  # num of features in V_2
        D = int(vector_feats_1.shape[-2])   # number of coordinate dimensions (D=3)

        # 1. Concatenate two features, 5D tensor
        vfeats = torch.cat((vector_feats_1, vector_feats_2), dim=-1)

        # 2. Linear combination (aka convolve), 5D tensor
        vfeats = torch.reshape(vfeats, shape=(-1, M1 + M2))
        vfeats = torch.matmul(vfeats, self.weight)
        if self.bias is not None:
            vfeats = vfeats + self.bias
        vfeats = torch.reshape(vfeats, shape=(-1, N, N, D, self.filters))

        # 3. Adjacency masking, 5D tensor
        adjacency = torch.reshape(adjacency, shape=(-1, N, N, 1, 1))
        adjacency = adjacency.repeat(1, 1, 1, D, self.filters)
        vfeats = vfeats * adjacency # element-wise multiplication

        # 4. Integrate over second atom axis, 4D tensor
        if self.pooling == "sum":
            vfeats = torch.sum(vfeats, dim=2)
        elif self.pooling == "mean":
            vfeats = torch.mean(vfeats, dim=2)
        elif self.pooling == "max":
            vfeats = torch.max(vfeats, dim=2)

        # 5. Activation, 4D tensor
        if self._activation is not None:
            vfeats = self._activation(vfeats)

        return vfeats

    def extra_repr(self) -> str:
        """String representation of the module. 
        
        Sets the extra representation, which comes into effect when 
        printing the (full) model summary.
        """
        summary = 'n_filters={_n_filters}, pooling={_pooling}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)