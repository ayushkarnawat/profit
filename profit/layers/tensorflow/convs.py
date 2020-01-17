"""Convolution operations for scalar and vector features."""

import tensorflow as tf

from keras import initializers, regularizers, activations
from keras.layers import Layer

from typing import Any, Dict, List, Optional, Tuple, Union


class GraphConvS(Layer):
    """Graph convolution for scalar features. 
    
    Each scalar feature :math: `s \\in \\mathbb{R}^M` describes the individual features for each 
    atom in the molecule. 
    
    Params:
    -------
    filters: int
        The number of filters/kernels to use. This is the number of atom-level features in a mol.

    pooling: str, optional, default="sum"
        Type of down-sample pooling technique to perform on the hidden representations after the 
        graph convolutions. The representations are either summed, averaged, or maxed depending 
        on the pooling type chosen.

    kernel_initializer: str, optional, default="glorot_uniform"
        Which statistical distribution (aka function) to use for the layer's kernel initial random 
        weights. See keras.initializers for list of all possible initializers.

    bias_initializer: str, optional, default="zeros"
        Which statistical distribution (aka function) to use for the layer's bias initial random 
        weights. See keras.initializers for list of all possible initializers.

    kernel_regularizer: str or None, optional, default=None
        Which optional regularizer to use to prevent overfitting. See keras.regulaizers for list 
        of all possible regularizers. Options include 'l1', 'l2', or 'l1_l2'. If 'None', then no 
        regularization is performed.

    activation: str or None, optional, default=None
        Which activation to apply to the hidden representations after pooling. See 
        keras.activations for list of all possible activation functions. If 'None', then no 
        activation is performed (aka linear activation), and the hidden representation is simply 
        returned. This is not recommened as non-linear activations model tasks better.  
    """

    def __init__(self,
                 filters: int,
                 pooling: Optional[str]="sum",
                 kernel_initializer: Optional[str]="glorot_uniform",
                 bias_initializer: Optional[str]="zeros",
                 kernel_regularizer: Optional[str]=None,
                 activation: Optional[str]=None,
                 **kwargs):
        # Check for valid params
        if filters > 0:
            self.filters = filters
        else:
            raise ValueError('Number of filters (N={0:d}) must be greater than 0.'.format(filters))

        if pooling == 'sum' or pooling == 'mean' or pooling == 'max':
            self.pooling = pooling
        else:
            raise ValueError("Pooling technique `{0:s}` is not valid. Available options are 'sum'"  
                             ",'mean', or 'max'.".format(pooling))
        
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activation = activations.get(activation)
        super(GraphConvS, self).__init__(**kwargs)


    def get_config(self) -> Dict[str, Any]:
        """Returns the configs of the layer.
        
        Returns:
        --------
        base_config: dict
            Contains all the configurations of the layer.
        """
        base_config = super(GraphConvS, self).get_config()
        base_config['filters'] = self.filters
        base_config['pooling'] = self.pooling
        return base_config


    def build(self, input_shape: Union[tf.Tensor, Union[List[tf.Tensor], Tuple[tf.Tensor, ...]]]) \
              -> None:
        """Create the layer's weight with the respective params defined above. 
        
        Params:
        -------
        input_shape: tf.Tensor or list/tuple of tf.Tensors
            Input tensor(s) to reference for weight shape computations.
        """
        M1 = input_shape[0][-1] # num of features in s_1
        M2 = input_shape[1][-1] # num of features in s_2
        self.w_conv_scalar = self.add_weight(shape=(M1 + M2, self.filters),
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             name='w_conv_scalar')

        self.b_conv_scalar = self.add_weight(shape=(self.filters,),
                                             name='b_conv_scalar',
                                             initializer=self.bias_initializer)
        super(GraphConvS, self).build(input_shape)


    def call(self, inputs: Union[tf.Tensor, Union[List[tf.Tensor], Tuple[tf.Tensor, ...]]], 
             mask: Union[tf.Tensor, Union[List[tf.Tensor], Tuple[tf.Tensor, ...]]]=None) \
             -> tf.Tensor:
        """Convolve two atom scalar features :math:`s_1, s_2`.
        
        More specifically, we (a) concatenate both scalars, (b) convolve features, (c) multiply 
        using the adjacency matrix, (d) downsample through pooling, and (e) apply a non-linear 
        activation.
        
        Params:
        -------
        inputs: tf.Tensor or list/tuple of tf.Tensors.
            Inputs to the current layer.

        mask: tf.Tensor, list/tuple of tf.Tensors, or None, optional, default=None
            Previous layer(s) that feed into this layer, if available. Not used.

        Returns:
        --------
        sfeats: tf.Tensor
            Output tensor after convolution, pooling, and activation. 
        """
        # Import graph tensors
        # scalar_feats_1 = (samples, max_atoms, max_atoms, num_atom_feats)
        # scalar_feats_2 = (samples, max_atoms, max_atoms, num_atom_feats)
        # adjacency = (samples, max_atoms, max_atoms)
        scalar_feats_1, scalar_feats_2, adjacency = inputs

        # Get parameters
        N = int(scalar_feats_1.shape[1])    # max number of atoms
        M1 = int(scalar_feats_1.shape[-1])  # num of features in s_1
        M2 = int(scalar_feats_2.shape[-1])  # num of features in s_2

        # 1. Concatenate two features, 4D tensor
        sfeats = tf.concat([scalar_feats_1, scalar_feats_2], axis=-1)

        # 2. Linear combination (aka convolve), 4D tensor
        sfeats = tf.reshape(sfeats, [-1, M1 + M2])
        sfeats = tf.matmul(sfeats, self.w_conv_scalar) + self.b_conv_scalar
        sfeats = tf.reshape(sfeats, [-1, N, N, self.filters])

        # 3. Adjacency masking, 4D tensor
        adjacency = tf.reshape(adjacency, [-1, N, N, 1])
        adjacency = tf.tile(adjacency, [1, 1, 1, self.filters])
        sfeats = tf.multiply(sfeats, adjacency)

        # 4. Integrate over second atom axis, 3D tensor
        if self.pooling == "sum":
            sfeats = tf.reduce_sum(sfeats, axis=2)
        elif self.pooling == "mean":
            sfeats = tf.reduce_mean(sfeats, axis=2)
        elif self.pooling == "max":
            sfeats = tf.reduce_max(sfeats, axis=2)

        # 5. Activation, 3D tensor
        sfeats = self.activation(sfeats)

        return sfeats


    def compute_output_shape(self, input_shape: Union[Tuple[int, ...], List[Tuple[int, ...]]]) \
                             -> Union[Tuple[int, ...], List[Tuple[int, ...]]]:
        """Compute the output shape of the layer.

        For the scalar convolution layer, the resulting shape will be: (k,N,M). Here, `N` is the 
        number of atoms, `M` is the number of atom-level features in a molecule, and `k` is the 
        number of samples in the batch/dataset.

        Params:
        -------
        input_shape: tuple of ints or list of tuples of ints
            Input shape of tensor(s).

        Returns:
        --------
        output_shape: tuple of ints or list of tuples of ints
            Output shape of tensor.
        """
        return input_shape[0][0], input_shape[0][1], self.filters


class GraphConvV(Layer):
    """Graph convolution for vector features. 
    
    Each vector feature :math: `V \\in \\mathbb{R}^{M \\times 3}` describes the vector features 
    (which include 3D for XYZ coords) for each atom in the molecule.
    
    Params:
    -------
    filters: int
        The number of filters/kernels to use.

    pooling: str, optional, default="sum"
        Type of down-sample pooling technique to perform on the hidden representations after the 
        graph convolutions. The representations are either summed, averaged, or maxed depending 
        on the pooling type chosen.

    kernel_initializer: str, optional, default="glorot_uniform"
        Which statistical distribution (aka function) to use for the layer's kernel initial random 
        weights. See keras.initializers for list of all possible initializers.

    bias_initializer: str, optional, default="zeros"
        Which statistical distribution (aka function) to use for the layer's bias initial random 
        weights. See keras.initializers for list of all possible initializers.

    kernel_regularizer: str or None, optional, default=None
        Which optional regularizer to use to prevent overfitting. See keras.regulaizers for list 
        of all possible regularizers. Options include 'l1', 'l2', or 'l1_l2'. If 'None', then no 
        regularization is performed.

    activation: str or None, optional, default=None
        Which activation to apply to the hidden representations after pooling. See 
        keras.activations for list of all possible activation functions. If 'None', then no 
        activation is performed (aka linear activation), and the hidden representation is simply 
        returned. This is not recommened as non-linear activations model tasks better.  
    """

    def __init__(self,
                 filters: int,
                 pooling: Optional[str]="sum",
                 kernel_initializer: Optional[str]="glorot_uniform",
                 bias_initializer: Optional[str]="zeros",
                 kernel_regularizer: Optional[str]=None,
                 activation: Optional[str]=None,
                 **kwargs):
        # Check for valid params
        if filters > 0:
            self.filters = filters
        else:
            raise ValueError('Number of filters (N={0:d}) must be greater than 0.'.format(filters))

        if pooling == 'sum' or pooling == 'mean' or pooling == 'max':
            self.pooling = pooling
        else:
            raise ValueError("Pooling technique `{0:s}` is not valid. Available options are 'sum'"  
                             ",'mean', or 'max'.".format(pooling))
        
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activation = activations.get(activation)
        super(GraphConvV, self).__init__(**kwargs)


    def get_config(self) -> Dict[str, Any]:
        """Returns the configs of the layer.
        
        Returns:
        --------
        base_config: dict
            Contains all the configurations of the layer.
        """
        base_config = super(GraphConvV, self).get_config()
        base_config['filters'] = self.filters
        base_config['pooling'] = self.pooling
        return base_config


    def build(self, input_shape: Union[tf.Tensor, Union[List[tf.Tensor], Tuple[tf.Tensor, ...]]]) \
              -> None:
        """Create the layer's weight with the respective params defined above. 
        
        Params:
        -------
        input_shape: tf.Tensor or list/tuple of tf.Tensors
            Input tensor(s) to reference for weight shape computations.
        """
        M1 = input_shape[0][-1] # num of features in V_1
        M2 = input_shape[1][-1] # num of features in V_2
        self.w_conv_vector = self.add_weight(shape=(M1 + M2, self.filters),
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             name='w_conv_vector')

        self.b_conv_vector = self.add_weight(shape=(self.filters,),
                                             initializer=self.bias_initializer,
                                             name='b_conv_vector')
        super(GraphConvV, self).build(input_shape)


    def call(self, inputs: Union[tf.Tensor, Union[List[tf.Tensor], Tuple[tf.Tensor, ...]]], 
             mask: Union[tf.Tensor, Union[List[tf.Tensor], Tuple[tf.Tensor, ...]]]=None) \
             -> tf.Tensor:
        """Convolve two atom vector features :math:`V_1, V_2`.
        
        More specifically, we (a) concatenate both vectors, (b) convolve features, (c) multiply 
        using the adjacency matrix, (d) downsample through pooling, and (e) apply a non-linear 
        activation.
        
        Params:
        -------
        inputs: tf.Tensor or list/tuple of tf.Tensors.
            Inputs to the current layer.

        mask: tf.Tensor, list/tuple of tf.Tensors, or None, optional, default=None
            Previous layer(s) that feed into this layer, if available. Not used.

        Returns:
        --------
        vfeats: tf.Tensor
            Output tensor after convolution, pooling, and activation. 
        """
        # Import graph tensors
        # vector_feats_1 = (samples, max_atoms, max_atoms, coor_dims, atom_feat)
        # vector_feats_2 = (samples, max_atoms, max_atoms, coor_dims, atom_feat)
        # adjacency = (samples, max_atoms, max_atoms)
        vector_feats_1, vector_feats_2, adjacency = inputs

        # Get parameters
        N = int(vector_feats_1.shape[1])    # max number of atoms
        M1 = int(vector_feats_1.shape[-1])  # num of features in V_1
        M2 = int(vector_feats_2.shape[-1])  # num of features in V_2
        D = int(vector_feats_1.shape[-2])   # number of coordinate dimensions (D=3) 

        # 1. Concatenate two features, 5D tensor
        vfeats = tf.concat([vector_feats_1, vector_feats_2], axis=-1)

        # 2. Linear combination (aka convolve), 5D tensor
        vfeats = tf.reshape(vfeats, [-1, M1 + M2])
        vfeats = tf.matmul(vfeats, self.w_conv_vector) + self.b_conv_vector
        vfeats = tf.reshape(vfeats, 
                                     [-1, N, N, D, self.filters])

        # 3. Adjacency masking, 5D tensor
        adjacency = tf.reshape(adjacency, [-1, N, N, 1, 1])
        adjacency = tf.tile(adjacency, [1, 1, 1, D, self.filters])
        vfeats = tf.multiply(vfeats, adjacency)

        # 4. Integrate over second atom axis, 4D tensor
        if self.pooling == "sum":
            vfeats = tf.reduce_sum(vfeats, axis=2)
        elif self.pooling == "mean":
            vfeats = tf.reduce_mean(vfeats, axis=2)
        elif self.pooling == "max":
            vfeats = tf.reduce_max(vfeats, axis=2)

        # 5. Activation, 4D tensor
        vfeats = self.activation(vfeats)

        return vfeats


    def compute_output_shape(self, input_shape: Union[Tuple[int, ...], List[Tuple[int, ...]]]) \
                             -> Union[Tuple[int, ...], List[Tuple[int, ...]]]:
        """Compute the output shape of the layer.

        For the vector convolution layer, the resulting shape will be: (k,N,3,M). Here, `N` is the 
        number of atoms, `M` is the number of atom-level features in a molecule, and `k` is the 
        number of samples in the batch/dataset.

        Params:
        -------
        input_shape: tuple of ints or list of tuples of ints
            Input shape of tensor(s).
        
        Returns:
        --------
        output_shape: tuple of ints or list of tuples of ints
            Output shape of tensor.
        """
        return input_shape[0][0], input_shape[0][1], input_shape[0][-2], self.filters
