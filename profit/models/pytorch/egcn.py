"""Three-dimensional embedded graph convolution network (eGCN).

TODO: Should each module contain a `output_shape()` function which 
determines the shape of the tensors after the forward pass is complete? 
This can be useful for defining the current layer's shape (similar to 
keras's `output_shape()` function).
TODO: Remove note that the `forward()` function assumes that the num  
samples are always batch_first. Change to support batch_last?

References:
- Three-Dimensionally Embedded Graph Convolutional Network
- Paper: https://arxiv.org/abs/1811.09794
- Code: https://github.com/blackmints/3DGCN
- Adapted from: https://git.io/Jv49v
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
    in_feats: int
        Input feature size.

    out_feats: int
        Output feature size (aka num of filters/kernels).

    pooling: str, optional, default="sum"
        Type of down-sample pooling technique to perform on the hidden 
        representations after the graph convolutions. Depending on the 
        pooling type chosen, the representations are etiher summed, 
        averaged, or maxed.

    bias: bool, default=True
        If True, adds a learnable bias to the convolved output.

    activation: callable activation function or None, optional, default=None
        If not None, applies an activation function to the updated node 
        features. See ::module:`torch.nn.modules.activation` for list 
        of all possible activation functions.

    Attributes
    ----------
    weight: torch.Tensor
        The learnable weight tensor.

    bias: torch.Tensor
        The learnable bias tensor.
    """

    def __init__(self, 
                 in_feats: int, 
                 out_feats: int, 
                 pooling: str="sum", 
                 bias: bool=True, 
                 activation: Optional[Activations]=None):
        super(GraphConvS, self).__init__()

        assert in_feats > 0, f"Number of input feats (N={in_feats}) must be greater than 0."
        assert out_feats > 0, f"Number of output feats (N={out_feats}) must be greater than 0."
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._pooling = pooling
        if activation is not None:
            assert isinstance(activation, tuple(activations)), f"Invalid " 
            f"activation func type: {type(activation)}. Should be one of the "
            f"following: {activations}."
        self._activation = activation

        # Add learnable weight/bias params to the layer
        self.weight = nn.Parameter(Tensor(in_feats, out_feats), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters() # init weights for 1st optim step


    def reset_parameters(self) -> None:
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
            Output tensor after computation.

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
        # NOTE: Assuming that the num_samples are in batch_first. 
        # Change to support batch_last?
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
        sfeats = torch.reshape(sfeats, shape=(-1, N, N, self._out_feats))

        # 3. Adjacency masking, 4D tensor
        adjacency = torch.reshape(adjacency, shape=(-1, N, N, 1))
        adjacency = adjacency.repeat(1, 1, 1, self._out_feats)
        sfeats = torch.mul(sfeats, adjacency) # element-wise multiplication

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
        summary = "in_feats={_in_feats}, out_feats={_out_feats}"
        summary += ", pooling={_pooling}"
        if "_activation" in self.__dict__:
            summary += ", activation={_activation}"
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
    in_feats: int
        Input feature size.

    out_feats: int
        Output feature size (aka num of filters/kernels).

    pooling: str, optional, default="sum"
        Type of down-sample pooling technique to perform on the hidden 
        representations after the graph convolutions. Depending on the 
        pooling type chosen, the representations are etiher summed, 
        averaged, or maxed.

    bias: bool, default=True
        If True, adds a learnable bias to the convolved output.

    activation: callable activation function or None, optional, default=None
        If not None, applies an activation function to the updated node 
        features. See ::module:`torch.nn.modules.activation` for list 
        of all possible activation functions.

    Attributes
    ----------
    weight: torch.Tensor
        The learnable weight tensor.

    bias: torch.Tensor
        The learnable bias tensor.
    """

    def __init__(self, 
                 in_feats: int,
                 out_feats: int, 
                 pooling: str="sum", 
                 bias: bool=True, 
                 activation: Optional[Activations]=None):
        super(GraphConvV, self).__init__()

        assert in_feats > 0, f"Number of input feats (N={in_feats}) must be greater than 0."
        assert out_feats > 0, f"Number of output feats (N={out_feats}) must be greater than 0."
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._pooling = pooling
        if activation is not None:
            assert isinstance(activation, tuple(activations)), f"Invalid " 
            f"activation func type: {type(activation)}. Should be one of the "
            f"following: {activations}."
        self._activation = activation

        # Add learnable weight/bias params to the layer
        self.weight = nn.Parameter(Tensor(in_feats, out_feats), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters() # init weights for 1st optim step


    def reset_parameters(self) -> None:
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
            Output tensor after computation.

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
        # NOTE: Assuming that the num_samples are in batch_first. 
        # Change to support batch_last?
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
        vfeats = torch.reshape(vfeats, shape=(-1, N, N, D, self._out_feats))

        # 3. Adjacency masking, 5D tensor
        adjacency = torch.reshape(adjacency, shape=(-1, N, N, 1, 1))
        adjacency = adjacency.repeat(1, 1, 1, D, self._out_feats)
        vfeats = torch.mul(vfeats, adjacency) # element-wise multiplication

        # 4. Integrate over second atom axis, 4D tensor
        if self._pooling == "sum":
            vfeats = torch.sum(vfeats, dim=2)
        elif self._pooling == "mean":
            vfeats = torch.mean(vfeats, dim=2)
        elif self._pooling == "max":
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
        summary = "in_feats={_in_feats}, out_feats={_out_feats}"
        summary += ", pooling={_pooling}"
        if "_activation" in self.__dict__:
            summary += ", activation={_activation}"
        return summary.format(**self.__dict__)


class GraphEmbed(nn.Module):
    """Joint graph embedding of atoms and their relative distances.

    NOTE: Although called an embeddings layer, the "embeddings" are not 
    trainable. Rather, it just processes the inputs to a common shape.
    """

    def __init__(self):
        super(GraphEmbed, self).__init__()


    def forward(self, inputs: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]) \
                -> List[torch.Tensor]:
        """Generate scalar and vector features for atoms and their 
        distances, respectively.

        Params:
        -------
        inputs: torch.Tensor or list/tuple of torch.Tensors.
            Inputs to the current module/layer.

        Returns:
        --------
        sfeats: torch.Tensor
            Atom scalar features.

        vfeats: torch.Tensor
            Atom vector features.
        """
        # NOTE: The scalar features (sfeats) are the computed atomic features itself
        # atoms = (samples, max_atoms, num_atom_feats)
        # distances = (samples, max_atoms, max_atoms, coor_dims)
        sfeats, distances = inputs

        # Get parameters
        # NOTE: Assuming that the num_samples are in batch_first. 
        # Change to support batch_last?
        max_atoms = int(sfeats.shape[1])
        num_atom_feats = int(sfeats.shape[-1])
        coor_dims = int(distances.shape[-1])

        # Generate vector feats filled with zeros, 4D tensor
        vfeats = torch.zeros_like(sfeats)
        vfeats = torch.reshape(vfeats, shape=(-1, max_atoms, 1, num_atom_feats))
        vfeats = vfeats.repeat(1, 1, coor_dims, 1)

        return [sfeats, vfeats]


class GraphGather(nn.Module):
    """Concatenates (pools) info across all atoms for each scalar and 
    vector feature in the graph.

    Allows the model to obtain more global information about the graph.

    Params:
    -------
    pooling: str, optional, default="sum"
        Type of down-sample pooling technique to perform on the hidden 
        representations after the graph convolutions. Depending on the 
        pooling type chosen, the representations are etiher summed, 
        averaged, or maxed.

    system: str, optional, default="cartesian"
        Whether to represent the data in spherical :math:`(r, \\theta, 
        \\phi)` or cartesian (XYZ) coordinates.

    activation: callable activation function or None, optional, default=None
        If not None, applies an activation function to the updated node 
        features. See ::module:`torch.nn.modules.activation` for list 
        of all possible activation functions.
    """

    def __init__(self, pooling: str="sum", system: str="cartesian", 
                 activation: Optional[Activations]=None):
        super(GraphGather, self).__init__()

        self._pooling = pooling
        self._system = system
        if activation is not None:
            assert isinstance(activation, tuple(activations)), f"Invalid " 
            f"activation func type: {type(activation)}. Should be one of the "
            f"following: {activations}."
        self._activation = activation


    def forward(self, inputs: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]) \
                -> List[torch.Tensor]:
        """Gather (concatenate) atom level information for the scalar 
        and vector features.

        More specifically, we (a) combine (integrate) over all the 
        features/information contained in the atom axis through pooling, 
        (b) apply a non-linear activation (if provided), and (c) map to 
        the specified coordinate system.

        Params:
        -------
        inputs: torch.Tensor or list/tuple of torch.Tensors.
            Inputs to the current module/layer.

        Returns:
        --------
        scalar_features: torch.Tensor
            Atom scalar features.

        vector_features: torch.Tensor
            Atom vector features.
        """
        # Import graph tensors
        # scalar_features = (samples, max_atoms, atom_feat)
        # vector_features = (samples, max_atoms, coor_dims, atom_feat)
        scalar_features, vector_features = inputs

        # Get parameters
        # NOTE: Assuming that the num_samples are in batch_first. 
        # Change to support batch_last?
        coor_dims = int(vector_features.shape[2])
        atom_feat = int(vector_features.shape[-1])

        # 1. Integrate over atom axis
        # TODO: Debug, this might be wrong (for vector features)
        if self.pooling == "sum":
            scalar_features = torch.sum(scalar_features, dim=1)
            vector_features = torch.sum(vector_features, dim=1)
        elif self.pooling == "mean":
            scalar_features = torch.mean(scalar_features, dim=1)
            vector_features = torch.mean(vector_features, dim=1)
        elif self.pooling == "max":
            scalar_features = torch.max(scalar_features, dim=1)
            # Select the vector feats that are max across all XYZ coordinates
            # NOTE: The reason we permute axis is so that we can use torch.gather along dim=-1
            vector_features = vector_features.permute(0, 2, 3, 1) # (samples, coor_dims, atom_feat, max_atoms)
            size = torch.sqrt(torch.sum(torch.square(vector_features), dim=1)) # (samples, atom_feat, max_atoms)
            # idxs of which atom has the highest value for that particular feature  
            idx = torch.reshape(torch.argmax(size, dim=-1), shape=(-1, 1, atom_feat, 1)) # (samples, 1, atom_feats, 1)
            idx = idx.repeat(1, coor_dims, 1, 1) # (samples, coor_dims, atom_feats, 1)
            vector_features = torch.reshape(torch.gather(vector_features, dim=-1, index=idx), 
                                            shape=(-1, coor_dims, atom_feat))

        # 2. Activation
        if self._activation is not None:
            scalar_features = self._activation(scalar_features) # (samples, atom_feat)
            vector_features = self._activation(vector_features) # (samples, coor_dims, atom_feat)

        # 3. Map to spherical coordinates, if specified
        if self._system == "spherical":
            x, y, z = torch.unbind(vector_features, dim=1)
            r = torch.sqrt(torch.square(x) + torch.square(y) + torch.square(z))
            # NOTE: We add 1 to all elements of x,r that are equal to 0 to avoid 
            # ZeroDivisionError. Additionally, we assume that the either r/z and 
            # x/y is of type torch.float
            t = torch.acos(z / (r + torch.eq(r,0).type(torch.float)))
            p = torch.atan(y / (x + torch.eq(x,0).type(torch.float)))
            vector_features = torch.stack([r, t, p], dim=1)

        return [scalar_features, vector_features]


    def extra_repr(self) -> str:
        """String representation of the module. 

        Sets the extra representation, which comes into effect when 
        printing the (full) model summary.
        """
        summary = "pooling={_pooling}, coord_system={_system}"
        if "_activation" in self.__dict__:
            summary += ", activation={_activation}"
        return summary.format(**self.__dict__)


class GraphSToS(nn.Module):
    """Scalar to scalar computation.

    Params:
    -------
    in_feats: int
        Input feature size.

    out_feats: int
        Output feature size (aka num of filters/kernels).

    bias: bool, default=True
        If True, adds a learnable bias to the convolved output.

    activation: callable activation function or None, optional, default=None
        If not None, applies an activation function to the updated node 
        features. See ::module:`torch.nn.modules.activation` for list 
        of all possible activation functions.

    Attributes:
    -----------
    weight: torch.Tensor
        The learnable weight tensor.

    bias: torch.Tensor
        The learnable bias tensor.
    """

    def __init__(self, 
                 in_feats: int, 
                 out_feats: int, 
                 bias: bool=True, 
                 activation: Optional[Activations]=None):
        super(GraphSToS, self).__init__()

        assert in_feats > 0, f"Number of input feats (N={in_feats}) must be greater than 0."
        assert out_feats > 0, f"Number of output feats (N={out_feats}) must be greater than 0."
        self._in_feats = in_feats
        self._out_feats = out_feats

        if activation is not None:
            assert isinstance(activation, tuple(activations)), f"Invalid " 
            f"activation func type: {type(activation)}. Should be one of the "
            f"following: {activations}."
        self._activation = activation

        # Add learnable weight/bias params to the layer
        self.weight = nn.Parameter(Tensor(in_feats, out_feats), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters() # init weights for 1st optim step


    def reset_parameters(self) -> None:
        """Reinitialize all learnable parameters."""
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)


    def forward(self, inputs: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]) \
                -> torch.Tensor:
        """Compute inter-atomic scalar to scalar features.

        More specifically, we (a) permute the scalar features for (b) 
        proper concatenation, (c) apply the learned weights from the 
        layer, and (d) apply a non-linear activation, if specified.

        Params:
        -------
        inputs: torch.Tensor or list/tuple of torch.Tensors.
            Inputs to the current layer.

        Returns:
        --------
        scalar_features: torch.Tensor
            Output tensor after computation.
        """
        # Import graph tensors
        # scalar_features = (samples, max_atoms, atom_feat)
        scalar_features = inputs

        # Get parameters
        # NOTE: Assuming that the num_samples are in batch_first. 
        # Change to support batch_last? 
        max_atoms = int(scalar_features.shape[1])
        atom_feat = int(scalar_features.shape[-1])

        # 1. Expand scalar features, 4D tensor
        scalar_features = torch.reshape(scalar_features, shape=(-1, max_atoms, 1, atom_feat))
        scalar_features = scalar_features.repeat(1, 1, max_atoms, 1)

        # 2. Combine between atoms, 4D tensor
        scalar_features_t = scalar_features.permute(0, 2, 1, 3)
        scalar_features = torch.cat((scalar_features, scalar_features_t), dim=-1)

        # 3. Apply weights, 4D tensor
        scalar_features = torch.reshape(scalar_features, shape=(-1, atom_feat * 2))
        scalar_features = torch.matmul(scalar_features, self.weight)
        if self.bias is not None:
            scalar_features = scalar_features + self.bias
        scalar_features = torch.reshape(scalar_features, shape=(-1, max_atoms, max_atoms, self._out_feats))

        # 4. Activation, 4D tensor
        if self._activation is not None:
            scalar_features = self._activation(scalar_features)

        return scalar_features


    def extra_repr(self) -> str:
        """String representation of the module. 

        Sets the extra representation, which comes into effect when 
        printing the (full) model summary.
        """
        summary = "in_feats={_in_feats}, out_feats={_out_feats}"
        if "_activation" in self.__dict__:
            summary += ", activation={_activation}"
        return summary.format(**self.__dict__)


class GraphSToV(nn.Module):
    """Scalar to vector computation.

    Params:
    -------
    in_feats: int
        Input feature size.

    out_feats: int
        Output feature size (aka num of filters/kernels).

    bias: bool, default=True
        If True, adds a learnable bias to the convolved output.

    activation: callable activation function or None, optional, default=None
        If not None, applies an activation function to the updated node 
        features. See ::module:`torch.nn.modules.activation` for list 
        of all possible activation functions.

    Attributes:
    -----------
    weight: torch.Tensor
        The learnable weight tensor.

    bias: torch.Tensor
        The learnable bias tensor.
    """

    def __init__(self, 
                 in_feats: int, 
                 out_feats: int, 
                 bias: bool=True, 
                 activation: Optional[Activations]=None):
        super(GraphSToV, self).__init__()

        assert in_feats > 0, f"Number of input feats (N={in_feats}) must be greater than 0."
        assert out_feats > 0, f"Number of output feats (N={out_feats}) must be greater than 0."
        self._in_feats = in_feats
        self._out_feats = out_feats

        if activation is not None:
            assert isinstance(activation, tuple(activations)), f"Invalid " 
            f"activation func type: {type(activation)}. Should be one of the "
            f"following: {activations}."
        self._activation = activation

        # Add learnable weight/bias params to the layer
        self.weight = nn.Parameter(Tensor(in_feats, out_feats), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters() # init weights for 1st optim step


    def reset_parameters(self) -> None:
        """Reinitialize all learnable parameters."""
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)


    def forward(self, inputs: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]) \
                -> torch.Tensor:
        """Compute atomic scalar to vector features.

        Specifically, we (a) expand the scalar features, (b) permute 
        them for proper concatenation, (c) apply the learned weights 
        from the layer, (d) multiply (element-wise) with the distance  
        matrix, and (e) apply a non-linear activation, if specified.

        Params:
        -------
        inputs: torch.Tensor or list/tuple of torch.Tensors.
            Inputs to the current layer.

        Returns:
        --------
        vector_features: torch.Tensor
            Output tensor after computation.
        """
        # Import graph tensors
        # scalar_features = (samples, max_atoms, atom_feat)
        # distances = (samples, max_atoms, max_atoms, coor_dims)
        scalar_features, distances = inputs

        # Get parameters
        max_atoms = int(scalar_features.shape[1])
        atom_feat = int(scalar_features.shape[-1])
        coor_dims = int(distances.shape[-1])

        # 1. Expand scalar features, 4D tensor
        scalar_features = torch.reshape(scalar_features, shape=(-1, max_atoms, 1, atom_feat))
        scalar_features = scalar_features.repeat(1, 1, max_atoms, 1)

        # 2. Combine between atoms, 4D tensor
        scalar_features_t = scalar_features.permute(0, 2, 1, 3)
        scalar_features = torch.cat((scalar_features, scalar_features_t), dim=-1)

        # 3. Apply weights, 5D tensor
        scalar_features = torch.reshape(scalar_features, shape=(-1, atom_feat * 2))
        scalar_features = torch.matmul(scalar_features, self.weight)
        if self.bias is not None:
            scalar_features = scalar_features + self.bias
        scalar_features = torch.reshape(scalar_features, shape=(-1, max_atoms, max_atoms, 1, self._out_feats))
        scalar_features = scalar_features.repeat(1, 1, 1, coor_dims, 1)

        # 4. Expand relative distances, 5D tensor
        distances = torch.reshape(distances, shape=(-1, max_atoms, max_atoms, coor_dims, 1))
        distances = distances.repeat(1, 1, 1, 1, self._out_feats)

        # 5. Tensor product, element-wise multiplication
        vector_features = torch.mul(scalar_features, distances)

        # 6. Activation
        if self._activation is not None:
            vector_features = self._activation(vector_features)

        return vector_features


    def extra_repr(self) -> str:
        """String representation of the module. 

        Sets the extra representation, which comes into effect when 
        printing the (full) model summary.
        """
        summary = "in_feats={_in_feats}, out_feats={_out_feats}"
        if "_activation" in self.__dict__:
            summary += ", activation={_activation}"
        return summary.format(**self.__dict__)


class GraphVToS(nn.Module):
    """Vector to scalar computation.

    Params:
    -------
    in_feats: int
        Input feature size.

    out_feats: int
        Output feature size (aka num of filters/kernels).

    bias: bool, default=True
        If True, adds a learnable bias to the convolved output.

    activation: callable activation function or None, optional, default=None
        If not None, applies an activation function to the updated node 
        features. See ::module:`torch.nn.modules.activation` for list 
        of all possible activation functions.

    Attributes:
    -----------
    weight: torch.Tensor
        The learnable weight tensor.

    bias: torch.Tensor
        The learnable bias tensor.
    """

    def __init__(self, 
                 in_feats: int, 
                 out_feats: int, 
                 bias: bool=True, 
                 activation: Optional[Activations]=None):
        super(GraphVToS, self).__init__()

        assert in_feats > 0, f"Number of input feats (N={in_feats}) must be greater than 0."
        assert out_feats > 0, f"Number of output feats (N={out_feats}) must be greater than 0."
        self._in_feats = in_feats
        self._out_feats = out_feats

        if activation is not None:
            assert isinstance(activation, tuple(activations)), f"Invalid " 
            f"activation func type: {type(activation)}. Should be one of the "
            f"following: {activations}."
        self._activation = activation

        # Add learnable weight/bias params to the layer
        self.weight = nn.Parameter(Tensor(in_feats, out_feats), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters() # init weights for 1st optim step


    def reset_parameters(self) -> None:
        """Reinitialize all learnable parameters."""
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)


    def forward(self, inputs: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]) \
                -> torch.Tensor:
        """Compute atomic vector to scalar features.

        Specifically, we (a) expand the vector features, (b) permute 
        them for proper concatenation, (c) apply the learned weights 
        from the layer, (d) project the vector onto r through (element-
        wise) multiplication, and (e) apply a non-linear activation, 
        if specified.

        Params:
        -------
        inputs: torch.Tensor or list/tuple of torch.Tensors.
            Inputs to the current layer.

        Returns:
        --------
        scalar_features: torch.Tensor
            Output tensor after computation.
        """
        # Import graph tensors
        # vector_features = (samples, max_atoms, coor_dims, atom_feat)
        # distances = (samples, max_atoms, max_atoms, coor_dims)
        vector_features, distances = inputs

        # Get parameters
        max_atoms = int(vector_features.shape[1])
        atom_feat = int(vector_features.shape[-1])
        coor_dims = int(vector_features.shape[-2])

        # 1. Expand vector features to 5D
        vector_features = torch.reshape(vector_features, shape=(-1, max_atoms, 1, coor_dims, atom_feat))
        vector_features = vector_features.repeat(1, 1, max_atoms, 1, 1)

        # 2. Combine between atoms
        vector_features_t = vector_features.permute(0, 2, 1, 3, 4)
        vector_features = torch.cat((vector_features, vector_features_t), dim=-1)

        # 3. Apply weights
        vector_features = torch.reshape(vector_features, shape=(-1, atom_feat * 2))
        vector_features = torch.matmul(vector_features, self.weight)
        if self.bias is not None:
            vector_features = vector_features + self.bias
        vector_features = torch.reshape(vector_features, shape=(-1, max_atoms, max_atoms, coor_dims, self._out_feats))

        # # 4. Calculate r^ = r / |r| and expand it to 5D
        # distances_hat = torch.sqrt(torch.sum(torch.square(distances), dim=-1, keepdim=True))
        # distances_hat = distances_hat + torch.eq(distances_hat, 0).type(torch.float)
        # distances_hat = torch.div(distances, distances_hat)
        # distances_hat = torch.reshape(distances_hat, shape=(-1, max_atoms, max_atoms, coor_dims, 1))
        # distances_hat = distances_hat.repeat(1, 1, 1, 1, self._out_feats)
        distances_hat = torch.reshape(distances, shape=(-1, max_atoms, max_atoms, coor_dims, 1))
        distances_hat = distances_hat.repeat(1, 1, 1, 1, self._out_feats)

        # 5. Projection of v onto r = v (dot) r^
        scalar_features = torch.mul(vector_features, distances_hat)
        scalar_features = torch.sum(scalar_features, dim=-2)

        # 6. Activation
        if self._activation is not None:
            scalar_features = self._activation(scalar_features)

        return scalar_features


    def extra_repr(self) -> str:
        """String representation of the module. 

        Sets the extra representation, which comes into effect when 
        printing the (full) model summary.
        """
        summary = "in_feats={_in_feats}, out_feats={_out_feats}"
        if "_activation" in self.__dict__:
            summary += ", activation={_activation}"
        return summary.format(**self.__dict__)


class GraphVToV(nn.Module):
    """Vector to vector computation.

    Params:
    -------
    in_feats: int
        Input feature size.

    out_feats: int
        Output feature size (aka num of filters/kernels).

    bias: bool, default=True
        If True, adds a learnable bias to the convolved output.

    activation: callable activation function or None, optional, default=None
        If not None, applies an activation function to the updated node 
        features. See ::module:`torch.nn.modules.activation` for list 
        of all possible activation functions.

    Attributes:
    -----------
    weight: torch.Tensor
        The learnable weight tensor.

    bias: torch.Tensor
        The learnable bias tensor.
    """

    def __init__(self, 
                 in_feats: int, 
                 out_feats: int, 
                 bias: bool=True, 
                 activation: Optional[Activations]=None):
        super(GraphVToV, self).__init__()

        assert in_feats > 0, f"Number of input feats (N={in_feats}) must be greater than 0."
        assert out_feats > 0, f"Number of output feats (N={out_feats}) must be greater than 0."
        self._in_feats = in_feats
        self._out_feats = out_feats

        if activation is not None:
            assert isinstance(activation, tuple(activations)), f"Invalid " 
            f"activation func type: {type(activation)}. Should be one of the "
            f"following: {activations}."
        self._activation = activation

        # Add learnable weight/bias params to the layer
        self.weight = nn.Parameter(Tensor(in_feats, out_feats), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters() # init weights for 1st optim step


    def reset_parameters(self) -> None:
        """Reinitialize all learnable parameters."""
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)


    def forward(self, inputs: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]) \
                -> torch.Tensor:
        """Compute atomic vector to vector features.

        Specifically, we (a) expand the vector features, (b) permute 
        them for proper concatenation, (c) apply the learned weights 
        from the layer, and (d) apply a non-linear activation, if 
        specified.

        Params:
        -------
        inputs: torch.Tensor or list/tuple of torch.Tensors.
            Inputs to the current layer.

        Returns:
        --------
        vector_features: torch.Tensor
            Output tensor after computation.
        """
        # Import graph tensors
        # vector_features = (samples, max_atoms, coor_dims, atom_feat)
        vector_features = inputs

        # Get parameters
        max_atoms = int(vector_features.shape[1])
        atom_feat = int(vector_features.shape[-1])
        coor_dims = int(vector_features.shape[-2])

        # 1. Expand vector features to 5D
        vector_features = torch.reshape(vector_features, shape=(-1, max_atoms, 1, coor_dims, atom_feat))
        vector_features = vector_features.repeat(1, 1, max_atoms, 1, 1)

        # 2. Combine between atoms
        vector_features_t = vector_features.permute(0, 2, 1, 3, 4)
        vector_features = torch.cat((vector_features, vector_features_t), dim=-1)

        # 3. Apply weights
        vector_features = torch.reshape(vector_features, shape=(-1, atom_feat * 2))
        vector_features = torch.matmul(vector_features, self.weight) 
        if self.bias is not None:
            vector_features = vector_features + self.bias
        vector_features = torch.reshape(vector_features, shape=(-1, max_atoms, max_atoms, coor_dims, self._out_feats))

        # 4. Activation
        if self._activation is not None:
            vector_features = self._activation(vector_features)

        return vector_features


    def extra_repr(self) -> str:
        """String representation of the module. 

        Sets the extra representation, which comes into effect when 
        printing the (full) model summary.
        """
        summary = "in_feats={_in_feats}, out_feats={_out_feats}"
        if "_activation" in self.__dict__:
            summary += ", activation={_activation}"
        return summary.format(**self.__dict__)


class EmbeddedGCN(nn.Module):
    """Assuming regression task. 
    
    TODO: Add L1/L2 regularizer for weight kernel on fully connected layers?
    """

    def __init__(self, num_atoms, num_feats, num_outputs, num_layers, units_conv, units_dense):
        super(Torch3DGCN, self).__init__()
        self.num_atoms = num_atoms
        self.num_feats = num_feats
        self.num_outputs = num_outputs
        self.num_layers = num_layers

        # Activations
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # Input layer
        self.embed = GraphEmbed()
        # Hidden layers
        self.hidden_layers = nn.ModuleDict()
        self.hidden_layers["s_to_s_0"] = GraphSToS(2*num_feats, units_conv, activation=self.relu)
        self.hidden_layers["v_to_s_0"] = GraphVToS(2*num_feats, units_conv, activation=self.relu)
        self.hidden_layers["s_to_v_0"] = GraphSToV(2*num_feats, units_conv, activation=self.tanh)
        self.hidden_layers["v_to_v_0"] = GraphVToV(2*num_feats, units_conv, activation=self.tanh)
        for i in range(1, num_layers):
            self.hidden_layers[f's_to_s_{i}'] = GraphSToS(2*units_conv, units_conv, activation=self.relu)
            self.hidden_layers[f'v_to_s_{i}'] = GraphVToS(2*units_conv, units_conv, activation=self.relu)
            self.hidden_layers[f's_to_v_{i}'] = GraphSToV(2*units_conv, units_conv, activation=self.relu)
            self.hidden_layers[f'v_to_v_{i}'] = GraphVToV(2*units_conv, units_conv, activation=self.relu)
        self.conv_s = GraphConvS(2*units_conv, units_conv, pooling="sum", activation=self.relu)
        self.conv_v = GraphConvV(2*units_conv, units_conv, pooling="sum", activation=self.tanh)
        # Gather layer
        self.gather = GraphGather(pooling="max")
        # Fully connected layers
        # NOTE: Applies dense connected layer to each coord (dim=1) of vector features (sample, coord_dim, n_filters/atom_feats)
        self.s_dense = nn.Linear(units_dense, units_dense)
        self.v_dense = nn.Linear(units_dense, units_dense)
        # Flatten layer
        self.flatten = nn.Flatten()
        # Output layer
        # NOTE: 3*units_dense for vector (since we flattened it) + units_dense from scalar = 512 total
        self.out = nn.Linear(3*units_dense+units_dense, num_outputs)


    def forward(self, inputs) -> torch.Tensor:
        # Input tensors
        # atoms = (samples, num_atoms, num_feats)
        # adjms = (samples, num_atoms, num_atoms)
        # dists = (samples, num_atoms, num_atoms, 3)
        atoms, adjms, dists = inputs

        sc, vc = self.embed([atoms, dists])
        for i in range(self.num_layers):
            sc_s = self.hidden_layers[f"s_to_s_{i}"](sc)
            sc_v = self.hidden_layers[f"v_to_s_{i}"]([vc, dists])
            vc_s = self.hidden_layers[f"s_to_v_{i}"]([sc, dists])
            vc_v = self.hidden_layers[f"v_to_v_{i}"](vc)
            sc = self.conv_s([sc_s, sc_v, adjms])
            vc = self.conv_v([vc_s, vc_v, adjms])
        sc, vc = self.gather([sc, vc])

        # apply relu activation to linear/dense unit(s)
        sc_out = self.relu(self.s_dense(sc))
        sc_out = self.relu(self.s_dense(sc_out))
        vc_out = self.relu(self.v_dense(vc))
        vc_out = self.relu(self.v_dense(vc_out))
        vc_out = self.flatten(vc_out)

        concat = torch.cat((sc_out, vc_out), dim=-1) # concatenate along last dimension
        return self.out(concat) # linear activation for regression task


    def extra_repr(self) -> str:
        pass