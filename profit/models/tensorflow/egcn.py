"""Three-dimensional embedded graph convolution network (eGCN).

References:
- Three-Dimensionally Embedded Graph Convolutional Network
- Paper: https://arxiv.org/abs/1811.09794
- Code: https://github.com/blackmints/3DGCN
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import tensorflow as tf

from keras import activations, initializers, regularizers
from keras.layers import Add, Concatenate, Dense, Dropout, Flatten, Input, Layer, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from profit.utils.training_utils.tensorflow.metrics import std_mae, std_rmse, std_r2


class GraphConvS(Layer):
    """Graph convolution for scalar features.

    Each scalar feature :math: `s \\in \\mathbb{R}^M` describes the 
    individual features for each atom in the molecule. 

    Params:
    -------
    filters: int
        The number of filters/kernels to use.

    pooling: str, optional, default="sum"
        Type of down-sample pooling technique to perform on the hidden 
        representations after the graph convolutions. The representations 
        are either summed, averaged, or maxed depending on the pooling 
        type chosen.

    kernel_initializer: str, optional, default="glorot_uniform"
        Which statistical distribution (aka function) to use for the 
        layer's kernel initial random weights. See keras.initializers 
        for list of all possible initializers.

    bias_initializer: str, optional, default="zeros"
        Which statistical distribution (aka function) to use for the 
        layer's bias initial random weights. See keras.initializers for 
        list of all possible initializers.

    kernel_regularizer: str or None, optional, default=None
        Which optional regularizer to use to prevent overfitting. See 
        keras.regulaizers for list of all possible regularizers. 
        Options include 'l1', 'l2', or 'l1_l2'. If 'None', then no 
        regularization is performed.

    activation: str or None, optional, default=None
        Which activation to apply to the hidden representations after 
        pooling. See keras.activations for list of all possible 
        activation functions. If 'None', then no activation is 
        performed (aka linear activation), and the hidden 
        representation is simply returned. This is not recommened as 
        non-linear activations model tasks better.   
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
        """Create the layer's weight with the params defined above.

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

        More specifically, we (a) concatenate both scalars, (b) convolve 
        features, (c) multiply using the adjacency matrix, (d) downsample 
        through pooling, and (e) apply a non-linear activation.

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

        For the scalar convolution layer, the resulting shape will be: 
        (k,N,M). Here, `N` is the number of atoms, `M` is the number of 
        atom-level features in a molecule, and `k` is the number of 
        samples in the batch/dataset.

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
    
    Each vector feature :math: `V \\in \\mathbb{R}^{M \\times 3}` 
    describes the vector features (which include 3D for XYZ coords) 
    for each atom in the molecule.
    
    Params:
    -------
    filters: int
        The number of filters/kernels to use.

    pooling: str, optional, default="sum"
        Type of down-sample pooling technique to perform on the hidden 
        representations after the graph convolutions. The representations 
        are either summed, averaged, or maxed depending on the pooling 
        type chosen.

    kernel_initializer: str, optional, default="glorot_uniform"
        Which statistical distribution (aka function) to use for the 
        layer's kernel initial random weights. See keras.initializers 
        for list of all possible initializers.

    bias_initializer: str, optional, default="zeros"
        Which statistical distribution (aka function) to use for the 
        layer's bias initial random weights. See keras.initializers for 
        list of all possible initializers.

    kernel_regularizer: str or None, optional, default=None
        Which optional regularizer to use to prevent overfitting. See 
        keras.regulaizers for list of all possible regularizers. 
        Options include 'l1', 'l2', or 'l1_l2'. If 'None', then no 
        regularization is performed.

    activation: str or None, optional, default=None
        Which activation to apply to the hidden representations after 
        pooling. See keras.activations for list of all possible 
        activation functions. If 'None', then no activation is 
        performed (aka linear activation), and the hidden 
        representation is simply returned. This is not recommened as 
        non-linear activations model tasks better.  
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
        """Create the layer's weight with the params defined above.

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

        More specifically, we (a) concatenate both vectors, (b) 
        convolve features, (c) multiply using the adjacency matrix, (d) 
        downsample through pooling, and (e) apply a non-linear activation.

        Params:
        -------
        inputs: tf.Tensor or list/tuple of tf.Tensors.
            Inputs to the current layer.

        mask: tf.Tensor, list/tuple of tf.Tensors, or None, optional, default=None
            Previous layer(s) that feed into this layer, if available. 
            Not used.

        Returns:
        --------
        vfeats: tf.Tensor
            Output tensor after convolution, pooling, and activation. 
        """
        # Input tensors
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
        vfeats = tf.reshape(vfeats, [-1, N, N, D, self.filters])

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

        For the vector convolution layer, the resulting shape will be: 
        (k,N,3,M). Here, `N` is the number of atoms, `M` is the number 
        of atom-level features in a molecule, and `k` is the number of 
        samples in the batch/dataset.

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


class GraphEmbed(Layer):
    """Joint graph embedding of atoms and their relative distances."""

    def __init__(self, **kwargs):
        super(GraphEmbed, self).__init__(**kwargs)

    
    def build(self, input_shape: Union[tf.Tensor, Union[List[tf.Tensor], Tuple[tf.Tensor, ...]]]) \
              -> None:
        """Create the layer's weights. 
        
        Params:
        -------
        input_shape: tf.Tensor or list/tuple of tf.Tensors
            Input tensor(s) to reference for weight shape computations.
        """
        super(GraphEmbed, self).build(input_shape)


    def call(self, inputs: Union[tf.Tensor, Union[List[tf.Tensor], Tuple[tf.Tensor, ...]]], 
             mask: Union[tf.Tensor, Union[List[tf.Tensor], Tuple[tf.Tensor, ...]]]=None) \
             -> tf.Tensor:
        """Generate scalar and vector features for atoms and their 
        distances, respectively.

        Params:
        -------
        inputs: tf.Tensor or list/tuple of tf.Tensors.
            Inputs to the current layer.

        mask: tf.Tensor, list/tuple of tf.Tensors, or None, optional, default=None
            Previous layer(s) that feed into this layer, if available. 
            Not used.

        Returns:
        --------
        sfeats: tf.Tensor
            Atom scalar features.

        vfeats: tf.Tensor
            Atom vector features.
        """
        # NOTE: The scalar features (sfeats) are the computed atomic features itself
        # atoms = (samples, max_atoms, num_atom_feats)
        # distances = (samples, max_atoms, max_atoms, coor_dims)
        sfeats, distances = inputs

        # Get params
        max_atoms = int(sfeats.shape[1])
        num_atom_feats = int(sfeats.shape[-1])
        coor_dims = int(distances.shape[-1])

        # Generate vector feats filled with zeros, 4D tensor
        vfeats = tf.zeros_like(sfeats)
        vfeats = tf.reshape(vfeats, [-1, max_atoms, 1, num_atom_feats])
        vfeats = tf.tile(vfeats, [1, 1, coor_dims, 1])

        return [sfeats, vfeats]

    
    def compute_output_shape(self, input_shape: Union[Tuple[int, ...], List[Tuple[int, ...]]]) \
                             -> List[Tuple[int, ...]]:
        """Compute the output shape of the layer.

        For the embedding layer, since we have both scalar and vector 
        features, we will have the following shapes: (a) sfeats.shape=(k,N,M), 
        and (b) vfeats.shape=(k,N,N,3). Here, `N` is the number of atoms, 
        `M` is the number of atom-level features in a molecule, and `k` is 
        the number of samples in the batch/dataset.

        Params:
        -------
        input_shape: tuple of ints or list of tuples of ints
            Input shape of tensor(s).

        Returns:
        --------
        output_shape: tuple of ints, or list of tuples of ints
            Output shape of tensor(s).
        """
        return [input_shape[0], 
                (input_shape[0][0], input_shape[0][1], input_shape[-1][-1], input_shape[0][-1])]


class GraphGather(Layer):
    """Combines both scalar and vector features into similar 
    representations. Essentially, like a concatination layer.
    
    Params:
    -------
    pooling: str, optional, default="sum"
        Type of down-sample pooling technique to perform on the hidden 
        representations after the graph convolutions. The representations 
        are either summed, averaged, or maxed depending on the pooling 
        type chosen.

    system: str, optional, default="cartesian"
        Whether to represent the data in spherical :math:
        `(r, \\theta, \\phi)` or cartesian (XYZ) coords.

    activation: str or None, optional, default=None
        Which activation to apply to the hidden representations after 
        pooling. See keras.activations for list of all possible 
        activation functions. If 'None', then no activation is performed 
        (aka linear activation), and the hidden representation is simply 
        returned. This is not recommened as non-linear activations model 
        tasks better.
    """
    def __init__(self,
                 pooling: Optional[str]="sum",
                 system: Optional[str]="cartesian",
                 activation: Optional[str]=None,
                 **kwargs):
        # Params check
        if pooling == 'sum' or pooling == 'mean' or pooling == 'max':
            self.pooling = pooling
        else:
            raise ValueError("Pooling technique `{0:s}` is not valid. Available options are 'sum'"  
                             ",'mean', or 'max'.".format(pooling))

        if system == 'spherical' or system == 'cartesian':
            self.system = system
        else:
            raise ValueError("Coordinate system must be 'cartesian' or 'spherical'.")

        self.activation = activations.get(activation)
        super(GraphGather, self).__init__(**kwargs)


    def get_config(self) -> Dict[str, Any]:
        """Returns the configs of the layer.

        Returns:
        --------
        base_config: dict
            Contains all the configurations of the layer.
        """
        base_config = super(GraphGather, self).get_config()
        base_config['pooling'] = self.pooling
        base_config['system'] = self.system
        return base_config


    def build(self, inputs_shape: Union[tf.Tensor, Union[List[tf.Tensor], Tuple[tf.Tensor, ...]]]) \
              -> None:
        """Create the layer's weight with the params defined above.

        Params:
        -------
        input_shape: tf.Tensor or list/tuple of tf.Tensors
            Input tensor(s) to reference for weight shape computations.
        """
        super(GraphGather, self).build(inputs_shape)


    def call(self, inputs: Union[tf.Tensor, List[tf.Tensor], Tuple[tf.Tensor, ...]], 
             mask: Optional[Union[tf.Tensor, List[tf.Tensor], Tuple[tf.Tensor, ...]]]=None) \
             -> tf.Tensor:
        """Gather and concatenate both scalar and vector features.

        More specifically, we (a) downsample both features through 
        pooling, (b) apply a non-linear activation, and (c) convert to 
        spherical coordinates, if needed.

        Params:
        -------
        inputs: tf.Tensor or list/tuple of tf.Tensors.
            Inputs to the current layer.

        mask: tf.Tensor, list/tuple of tf.Tensors, or None, optional, default=None
            Previous layer(s) that feed into this layer, if available. 
            Not used.

        Returns:
        --------
        scalar features: tf.Tensor
            Atom scalar features.

        vector features: tf.Tensor
            Atom vector features.
        """
        # Import graph tensors
        # scalar_features = (samples, max_atoms, atom_feat)
        # vector_features = (samples, max_atoms, coor_dims, atom_feat)
        scalar_features, vector_features = inputs

        # Get parameters
        coor_dims = int(vector_features.shape[2])
        atom_feat = int(vector_features.shape[-1])

        # 1. Integrate over atom axis
        # TODO(ayush): Debug, this might be wrong
        if self.pooling == "sum":
            scalar_features = tf.reduce_sum(scalar_features, axis=1)
            vector_features = tf.reduce_sum(vector_features, axis=1)
        elif self.pooling == "mean":
            scalar_features = tf.reduce_mean(scalar_features, axis=1)
            vector_features = tf.reduce_mean(vector_features, axis=1)
        elif self.pooling == "max":
            scalar_features = tf.reduce_max(scalar_features, axis=1)

            vector_features = tf.transpose(vector_features, perm=[0, 2, 3, 1]) # (samples, coor_dims, atom_feat, max_atoms)
            size = tf.sqrt(tf.reduce_sum(tf.square(vector_features), axis=1)) # (samples, atom_feat, max_atoms)
            # idxs of which atom has the best/highest value for that particular feature  
            idx = tf.reshape(tf.argmax(size, axis=-1, output_type=tf.int32), 
                             [-1, 1, atom_feat, 1]) # (samples, 1, atom_feats, 1)
            idx = tf.tile(idx, [1, coor_dims, 1, 1]) # (samples, coor_dims, atom_feats, 1)
            vector_features = tf.reshape(tf.batch_gather(vector_features, idx), 
                                         [-1, coor_dims, atom_feat])

        # 2. Activation
        scalar_features = self.activation(scalar_features) # (samples, atom_feat)
        vector_features = self.activation(vector_features) # (samples, coor_dims, atom_feat)

        # 3. Conversion, if specified
        if self.system == "spherical":
            x, y, z = tf.unstack(vector_features, axis=1)
            r = tf.sqrt(tf.square(x) + tf.square(y) + tf.square(z))
            t = tf.acos(tf.divide(z, r + tf.cast(tf.equal(r, 0), dtype=float)))
            p = tf.atan(tf.divide(y, x + tf.cast(tf.equal(x, 0), dtype=float)))
            vector_features = tf.stack([r, t, p], axis=1)

        return [scalar_features, vector_features]


    def compute_output_shape(self, inputs_shape: Union[Tuple[int, ...], List[Tuple[int, ...]]]) \
                             -> Union[Tuple[int, ...], List[Tuple[int, ...]]]:
        """Compute the output shape of the layer.

        For the gather layer, since we have both scalar and vector 
        features, we will have the following shapes: (a) sfeats.shape=(k,M), 
        and (b) vfeats.shape=(k,3,M). Here, `M` is the number of atom-
        level features in a molecule, and `k` is the number of samples 
        in the batch/dataset.

        Params:
        -------
        inputs_shape: tuple of ints or list of tuples of ints
            Input shape of tensor(s).

        Returns:
        --------
        output_shape: tuple of ints, or list of tuples of ints
            Output shape of tensor(s).
        """
        return [(inputs_shape[0][0], inputs_shape[0][2]), 
                (inputs_shape[1][0], inputs_shape[1][2], inputs_shape[1][3])]


class GraphSToS(Layer):
    """Scalar to scalar computation.

    Params:
    -------
    filters: int
        The number of filters/kernels to use. This is the number of 
        atom-level features in a mol.

    kernel_initializer: str, optional, default="glorot_uniform"
        Which statistical distribution (aka function) to use for the 
        layer's kernel initial random weights. See keras.initializers 
        for list of all possible initializers.

    bias_initializer: str, optional, default="zeros"
        Which statistical distribution (aka function) to use for the 
        layer's bias initial random weights. See keras.initializers for 
        list of all possible initializers.

    kernel_regularizer: str or None, optional, default=None
        Which optional regularizer to use to prevent overfitting. See 
        keras.regulaizers for list of all possible regularizers. Options 
        include 'l1', 'l2', or 'l1_l2'. If 'None', then no regularization 
        is performed.

    activation: str or None, optional, default=None
        Which activation to apply to the hidden representations after 
        pooling. See keras.activations for list of all possible 
        activation functions. If 'None', then no activation is performed 
        (aka linear activation), and the hidden representation is simply 
        returned. This is not recommened as non-linear activations model 
        tasks better.  
    """

    def __init__(self,
                 filters,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 activation=None,
                 **kwargs):
        # Check for valid params
        if filters > 0:
            self.filters = filters
        else:
            raise ValueError('Number of filters (N={0:d}) must be greater than 0.'.format(filters))

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activation = activations.get(activation)
        super(GraphSToS, self).__init__(**kwargs)


    def get_config(self) -> Dict[str, Any]:
        """Returns the configs of the layer.
        
        Returns:
        --------
        base_config: dict
            Contains all the configurations of the layer.
        """
        base_config = super(GraphSToS, self).get_config()
        base_config['filters'] = self.filters
        return base_config


    def build(self, input_shape: Union[tf.Tensor, Union[List[tf.Tensor], Tuple[tf.Tensor, ...]]]) \
              -> None:
        """Create the layer's weight with the respective params defined above. 
        
        Params:
        -------
        input_shape: tf.Tensor or list/tuple of tf.Tensors
            Input tensor(s) to reference for weight shape computations.
        """
        atom_feat = input_shape[-1]
        self.w_ss = self.add_weight(shape=(atom_feat * 2, self.filters),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    name='w_ss')

        self.b_ss = self.add_weight(shape=(self.filters,),
                                    name='b_ss',
                                    initializer=self.bias_initializer)

        super(GraphSToS, self).build(input_shape)


    def call(self, inputs: Union[tf.Tensor, Union[List[tf.Tensor], Tuple[tf.Tensor, ...]]], 
             mask: Union[tf.Tensor, Union[List[tf.Tensor], Tuple[tf.Tensor, ...]]]=None) \
             -> tf.Tensor:
        """Compute inter-atomic scalar to scalar features.
        
        More specifically, we (a) transpose the scalar features for (b) proper concatenation, (c) 
        apply the learned weights from the layer, and (d) apply a non-linear activation.
        
        Params:
        -------
        inputs: tf.Tensor or list/tuple of tf.Tensors.
            Inputs to the current layer.

        mask: tf.Tensor, list/tuple of tf.Tensors, or None, optional, default=None
            Previous layer(s) that feed into this layer, if available. Not used.

        Returns:
        --------
        scalar_features: tf.Tensor
            Output tensor after convolution, pooling, and activation. 
        """
        # Import graph tensors
        # scalar_features = (samples, max_atoms, atom_feat)
        scalar_features = inputs

        # Get parameters
        max_atoms = int(scalar_features.shape[1])
        atom_feat = int(scalar_features.shape[-1])

        # 1. Expand scalar features, 4D tensor
        scalar_features = tf.reshape(scalar_features, [-1, max_atoms, 1, atom_feat])
        scalar_features = tf.tile(scalar_features, [1, 1, max_atoms, 1])

        # 2. Combine between atoms, 4D tensor
        scalar_features_t = tf.transpose(scalar_features, perm=[0, 2, 1, 3])
        scalar_features = tf.concat([scalar_features, scalar_features_t], -1)

        # 3. Apply weights, 4D tensor
        scalar_features = tf.reshape(scalar_features, [-1, atom_feat * 2])
        scalar_features = tf.matmul(scalar_features, self.w_ss) + self.b_ss
        scalar_features = tf.reshape(scalar_features, [-1, max_atoms, max_atoms, self.filters])

        # 4. Activation, 4D tensor
        scalar_features = self.activation(scalar_features)

        return scalar_features


    def compute_output_shape(self, input_shape: Union[Tuple[int, ...], List[Tuple[int, ...]]]) \
                             -> Union[Tuple[int, ...], List[Tuple[int, ...]]]:
        """Compute the output shape of the layer.

        For the scalar to scalar computation layer, the resulting shape will be: (k,N,N,M). Here, 
        `N` is the number of atoms, `M` is the number of atom-level features in a molecule, and 
        `k` is the number of samples in the batch/dataset.

        Params:
        -------
        input_shape: tuple of ints or list of tuples of ints
            Input shape of tensor(s).

        Returns:
        --------
        output_shape: tuple of ints, or list of tuples of ints
            Output shape of tensor(s).
        """
        return input_shape[0], input_shape[1], input_shape[1], self.filters


class GraphSToV(Layer):
    """Scalar to vector computation.

    Params:
    -------
    filters: int
        The number of filters/kernels to use. This is the number of atom-level features in a mol.

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
                 filters,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 activation=None,
                 **kwargs):
        # Check for valid params
        if filters > 0:
            self.filters = filters
        else:
            raise ValueError('Number of filters (N={0:d}) must be greater than 0.'.format(filters))

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activation = activations.get(activation)
        super(GraphSToV, self).__init__(**kwargs)


    def get_config(self) -> Dict[str, Any]:
        """Returns the configs of the layer.
        
        Returns:
        --------
        base_config: dict
            Contains all the configurations of the layer.
        """
        base_config = super(GraphSToV, self).get_config()
        base_config['filters'] = self.filters
        return base_config


    def build(self, input_shape: Union[tf.Tensor, Union[List[tf.Tensor], Tuple[tf.Tensor, ...]]]) \
              -> None:
        """Create the layer's weight with the respective params defined above. 
        
        Params:
        -------
        input_shape: tf.Tensor or list/tuple of tf.Tensors
            Input tensor(s) to reference for weight shape computations.
        """
        atom_feat = input_shape[0][-1]
        self.w_sv = self.add_weight(shape=(atom_feat * 2, self.filters),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    name='w_sv')

        self.b_sv = self.add_weight(shape=(self.filters,),
                                    name='b_sv',
                                    initializer=self.bias_initializer)

        super(GraphSToV, self).build(input_shape)


    def call(self, inputs: Union[tf.Tensor, Union[List[tf.Tensor], Tuple[tf.Tensor, ...]]], 
             mask: Union[tf.Tensor, Union[List[tf.Tensor], Tuple[tf.Tensor, ...]]]=None) \
             -> tf.Tensor:
        """Compute atomic scalar to vector features.
        
        More specifically, we (a) expand the scalar features, (b) transpose them for proper 
        concatenation, (c) apply the learned weights from the layer, (d) multiply with the relative 
        distance matrix, and (e) apply a non-linear activation.
        
        Params:
        -------
        inputs: tf.Tensor or list/tuple of tf.Tensors.
            Inputs to the current layer.

        mask: tf.Tensor, list/tuple of tf.Tensors, or None, optional, default=None
            Previous layer(s) that feed into this layer, if available. Not used.

        Returns:
        --------
        scalar_features: tf.Tensor
            Output tensor after convolution, pooling, and activation. 
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
        scalar_features = tf.reshape(scalar_features, [-1, max_atoms, 1, atom_feat])
        scalar_features = tf.tile(scalar_features, [1, 1, max_atoms, 1])

        # 2. Combine between atoms, 4D tensor
        scalar_features_t = tf.transpose(scalar_features, perm=[0, 2, 1, 3])
        scalar_features = tf.concat([scalar_features, scalar_features_t], -1)

        # 3. Apply weights, 5D tensor
        scalar_features = tf.reshape(scalar_features, [-1, atom_feat * 2])
        scalar_features = tf.matmul(scalar_features, self.w_sv) + self.b_sv
        scalar_features = tf.reshape(scalar_features, [-1, max_atoms, max_atoms, 1, self.filters])
        scalar_features = tf.tile(scalar_features, [1, 1, 1, coor_dims, 1])

        # 4. Expand relative distances, 5D tensor
        distances = tf.reshape(distances, [-1, max_atoms, max_atoms, coor_dims, 1])
        distances = tf.tile(distances, [1, 1, 1, 1, self.filters])

        # 5. Tensor product
        vector_features = tf.multiply(scalar_features, distances)

        # 6. Activation
        vector_features = self.activation(vector_features)

        return vector_features


    def compute_output_shape(self, input_shape: Union[Tuple[int, ...], List[Tuple[int, ...]]]) \
                             -> List[Tuple[int, ...]]:
        """Compute the output shape of the layer.

        For the scalar to vector computation layer, the resulting shape will be: (k,N,N,3,M). 
        Here, `N` is the number of atoms, `M` is the number of atom-level features in a molecule, 
        and `k` is the number of samples in the batch/dataset.

        Params:
        -------
        input_shape: tuple of ints or list of tuples of ints
            Input shape of tensor(s).

        Returns:
        --------
        output_shape: tuple of ints, or list of tuples of ints
            Output shape of tensor(s).
        """
        return input_shape[0][0], input_shape[0][1], input_shape[0][1], input_shape[1][-1], self.filters


class GraphVToV(Layer):
    """Vector to vector computation.
    
    Params:
    -------
    filters: int
        The number of filters/kernels to use. This is the number of atom-level features in a mol.

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
                 filters,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 activation=None,
                 **kwargs):
        # Check for valid params
        if filters > 0:
            self.filters = filters
        else:
            raise ValueError('Number of filters (N={0:d}) must be greater than 0.'.format(filters))

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activation = activations.get(activation)
        super(GraphVToV, self).__init__(**kwargs)


    def get_config(self) -> Dict[str, Any]:
        """Returns the configs of the layer.
        
        Returns:
        --------
        base_config: dict
            Contains all the configurations of the layer.
        """
        base_config = super(GraphVToV, self).get_config()
        base_config['filters'] = self.filters
        return base_config


    def build(self, input_shape: Union[tf.Tensor, Union[List[tf.Tensor], Tuple[tf.Tensor, ...]]]) \
              -> None:
        """Create the layer's weight with the respective params defined above. 
        
        Params:
        -------
        input_shape: tf.Tensor or list/tuple of tf.Tensors
            Input tensor(s) to reference for weight shape computations.
        """
        atom_feat = input_shape[-1]
        self.w_vv = self.add_weight(shape=(atom_feat * 2, self.filters),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    name='w_vv')

        self.b_vv = self.add_weight(shape=(self.filters,),
                                    name='b_vv',
                                    initializer=self.bias_initializer)

        super(GraphVToV, self).build(input_shape)


    def call(self, inputs: Union[tf.Tensor, Union[List[tf.Tensor], Tuple[tf.Tensor, ...]]], 
             mask: Union[tf.Tensor, Union[List[tf.Tensor], Tuple[tf.Tensor, ...]]]=None) \
             -> tf.Tensor:
        """Compute atomic vector to vector features.
        
        More specifically, we (a) expand the vector features, (b) transpose them for proper 
        concatenation, (c) apply the learned weights from the layer, and (d) apply a non-linear 
        activation.
        
        Params:
        -------
        inputs: tf.Tensor or list/tuple of tf.Tensors.
            Inputs to the current layer.

        mask: tf.Tensor, list/tuple of tf.Tensors, or None, optional, default=None
            Previous layer(s) that feed into this layer, if available. Not used.

        Returns:
        --------
        scalar_features: tf.Tensor
            Output tensor after convolution, pooling, and activation. 
        """
        # Import graph tensors
        # vector_features = (samples, max_atoms, coor_dims, atom_feat)
        vector_features = inputs

        # Get parameters
        max_atoms = int(vector_features.shape[1])
        atom_feat = int(vector_features.shape[-1])
        coor_dims = int(vector_features.shape[-2])

        # 1. Expand vector features to 5D
        vector_features = tf.reshape(vector_features, [-1, max_atoms, 1, coor_dims, atom_feat])
        vector_features = tf.tile(vector_features, [1, 1, max_atoms, 1, 1])

        # 2. Combine between atoms
        vector_features_t = tf.transpose(vector_features, perm=[0, 2, 1, 3, 4])
        vector_features = tf.concat([vector_features, vector_features_t], -1)

        # 3. Apply weights
        vector_features = tf.reshape(vector_features, [-1, atom_feat * 2])
        vector_features = tf.matmul(vector_features, self.w_vv) + self.b_vv
        vector_features = tf.reshape(vector_features, [-1, max_atoms, max_atoms, coor_dims, self.filters])

        # 4. Activation
        vector_features = self.activation(vector_features)

        return vector_features

    def compute_output_shape(self, input_shape: Union[Tuple[int, ...], List[Tuple[int, ...]]]) \
                             -> List[Tuple[int, ...]]:
        """Compute the output shape of the layer.

        For the vector to vector computation layer, the resulting shape will be: (k,N,N,3,M). 
        Here, `N` is the number of atoms, `M` is the number of atom-level features in a molecule, 
        and `k` is the number of samples in the batch/dataset.

        Params:
        -------
        input_shape: tuple of ints or list of tuples of ints
            Input shape of tensor(s).

        Returns:
        --------
        output_shape: tuple of ints, or list of tuples of ints
            Output shape of tensor(s).
        """
        return input_shape[0], input_shape[1], input_shape[1], input_shape[-2], self.filters


class GraphVToS(Layer):
    """Vector to scalar computation.
    
    Params:
    -------
    filters: int
        The number of filters/kernels to use. This is the number of atom-level features in a mol.

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
                 filters,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 activation=None,
                 **kwargs):
        # Check for valid params
        if filters > 0:
            self.filters = filters
        else:
            raise ValueError('Number of filters (N={0:d}) must be greater than 0.'.format(filters))

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activation = activations.get(activation)
        super(GraphVToS, self).__init__(**kwargs)


    def get_config(self) -> Dict[str, Any]:
        """Returns the configs of the layer.
        
        Returns:
        --------
        base_config: dict
            Contains all the configurations of the layer.
        """
        base_config = super(GraphVToS, self).get_config()
        base_config['filters'] = self.filters
        return base_config


    def build(self, input_shape: Union[tf.Tensor, Union[List[tf.Tensor], Tuple[tf.Tensor, ...]]]) \
              -> None:
        """Create the layer's weight with the respective params defined above. 
        
        Params:
        -------
        input_shape: tf.Tensor or list/tuple of tf.Tensors
            Input tensor(s) to reference for weight shape computations.
        """
        atom_feat = input_shape[0][-1]
        self.w_vs = self.add_weight(shape=(atom_feat * 2, self.filters),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    name='w_vs')

        self.b_vs = self.add_weight(shape=(self.filters,),
                                    name='b_vs',
                                    initializer=self.bias_initializer)

        super(GraphVToS, self).build(input_shape)


    def call(self, inputs: Union[tf.Tensor, Union[List[tf.Tensor], Tuple[tf.Tensor, ...]]], 
             mask: Union[tf.Tensor, Union[List[tf.Tensor], Tuple[tf.Tensor, ...]]]=None) \
             -> tf.Tensor:
        """Compute atomic vector to scalar features.
        
        More specifically, we (a) expand the vector features, (b) transpose them for proper 
        concatenation, (c) apply the learned weights from the layer, (d) project vector onto r 
        through multiplication, and (e) apply a non-linear activation.
        
        Params:
        -------
        inputs: tf.Tensor or list/tuple of tf.Tensors.
            Inputs to the current layer.

        mask: tf.Tensor, list/tuple of tf.Tensors, or None, optional, default=None
            Previous layer(s) that feed into this layer, if available. Not used.

        Returns:
        --------
        scalar_features: tf.Tensor
            Output tensor after convolution, pooling, and activation. 
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
        vector_features = tf.reshape(vector_features, [-1, max_atoms, 1, coor_dims, atom_feat])
        vector_features = tf.tile(vector_features, [1, 1, max_atoms, 1, 1])

        # 2. Combine between atoms
        vector_features_t = tf.transpose(vector_features, perm=[0, 2, 1, 3, 4])
        vector_features = tf.concat([vector_features, vector_features_t], -1)

        # 3. Apply weights
        vector_features = tf.reshape(vector_features, [-1, atom_feat * 2])
        vector_features = tf.matmul(vector_features, self.w_vs) + self.b_vs
        vector_features = tf.reshape(vector_features, [-1, max_atoms, max_atoms, coor_dims, self.filters])

        # 4. Calculate r^ = r / |r| and expand it to 5D
        # distances_hat = tf.sqrt(tf.reduce_sum(tf.square(distances), axis=-1, keepdims=True))
        # distances_hat = distances_hat + tf.cast(tf.equal(distances_hat, 0), tf.float32)
        # distances_hat = tf.divide(distances, distances_hat)
        # distances_hat = tf.reshape(distances_hat, [-1, max_atoms, max_atoms, coor_dims, 1])
        # distances_hat = tf.tile(distances_hat, [1, 1, 1, 1, self.filters])
        distances_hat = tf.reshape(distances, [-1, max_atoms, max_atoms, coor_dims, 1])
        distances_hat = tf.tile(distances_hat, [1, 1, 1, 1, self.filters])

        # 5. Projection of v onto r = v (dot) r^
        scalar_features = tf.multiply(vector_features, distances_hat)
        scalar_features = tf.reduce_sum(scalar_features, axis=-2)

        # 6. Activation
        scalar_features = self.activation(scalar_features)

        return scalar_features


    def compute_output_shape(self, input_shape: Union[Tuple[int, ...], List[Tuple[int, ...]]]) \
                             -> Union[Tuple[int, ...], List[Tuple[int, ...]]]:
        """Compute the output shape of the layer.

        For the vector to scalar computation layer, the resulting shape will be: (k,N,N,M). Here, 
        `N` is the number of atoms, `M` is the number of atom-level features in a molecule, and 
        `k` is the number of samples in the batch/dataset.

        Params:
        -------
        input_shape: tuple of ints or list of tuples of ints
            Input shape of tensor(s).

        Returns:
        --------
        output_shape: tuple of ints, or list of tuples of ints
            Output shape of tensor(s).
        """
        return input_shape[0][0], input_shape[0][1], input_shape[0][1], self.filters


class EmbeddedGCN(object):
    """Embedded Graph Convolution Network.

    Params:
    -------
    num_atoms: int
        Max number of atoms in a molecule.

    num_feats: int
        Number of features per atom in the molecule.

    num_outputs: int
        Number of outputs. This is simply the number of unique labels.

    units_conv: int, optional, default=128
        Number of filters/units to use in the convolution layer(s).

    units_dense: int, optional, default=128
        Number of filters/units to use in the fully connected layer(s).

    num_layers: int, optional, default=2
        Total number of graph convolution layer(s).

    task: str, optional, default='regression'
        Whether the task is classification (binary or multi-label) or 
        regression-based. 

    loss: str, optional, default="mse"
        Loss function. See keras.losses for list of all available losses.

    pooling: str, optional, default="max"
        Type of down-sample pooling technique to perform on the hidden 
        representations after the graph convolutions. The representations 
        are either summed, averaged, or maxed depending on the pooling 
        type chosen.

    std: float or None, optional, default=None
        Standard deviation of the dataset. If regression-based task, 
        then a std must be provided, which is usually the standard 
        deviation of the training dataset.
    """

    def __init__(self, num_atoms: int, num_feats: int, num_outputs: int,
                 units_conv: Optional[int]=128, units_dense: Optional[int]=128, 
                 num_layers: Optional[int]=2, task: Optional[str]="regression",
                 loss: Optional[str]="mse", pooling: Optional[str]="max", 
                 std: Optional[float]=None):
        self.num_atoms = num_atoms
        self.num_feats = num_feats
        self.num_outputs = num_outputs
        self.units_conv = units_conv
        self.units_dense = units_dense
        self.num_layers = num_layers
        self.task = task
        self.loss = loss
        self.pooling = pooling

        if self.task == 'regression':
            if std is not None:
                self.std = std
            else:
                raise ValueError("Standard deviation of the dataset must be "
                                 "provided for regression tasks.")

    def get_model(self) -> Model:
        """Retrive the model with the specified params.

        Returns:
        --------
        model: keras.models.Model
            The compiled model.
        """
        atoms = Input(name='atom_inputs', shape=(self.num_atoms, self.num_feats))
        adjms = Input(name='adjm_inputs', shape=(self.num_atoms, self.num_atoms))
        dists = Input(name='coor_inputs', shape=(self.num_atoms, self.num_atoms, 3))

        sc, vc = GraphEmbed()([atoms, dists])

        for _ in range(self.num_layers):
            sc_s = GraphSToS(self.units_conv, activation='relu')(sc)
            sc_v = GraphVToS(self.units_conv, activation='relu')([vc, dists])

            vc_s = GraphSToV(self.units_conv, activation='tanh')([sc, dists])
            vc_v = GraphVToV(self.units_conv, activation='tanh')(vc)

            sc = GraphConvS(self.units_conv, pooling='sum', activation='relu')([sc_s, sc_v, adjms])
            vc = GraphConvV(self.units_conv, pooling='sum', activation='tanh')([vc_s, vc_v, adjms])

        sc, vc = GraphGather(pooling=self.pooling)([sc, vc])
        sc_out = Dense(self.units_dense, activation='relu', kernel_regularizer=l2(0.005))(sc)
        sc_out = Dense(self.units_dense, activation='relu', kernel_regularizer=l2(0.005))(sc_out)

        # Apply a dense activation layer to each vector feature
        vc_out = TimeDistributed(Dense(self.units_dense, activation='relu', 
                                       kernel_regularizer=l2(0.005)))(vc)
        vc_out = TimeDistributed(Dense(self.units_dense, activation='relu', 
                                       kernel_regularizer=l2(0.005)))(vc_out)
        vc_out = Flatten()(vc_out)

        out = Concatenate(axis=-1)([sc_out, vc_out])

        if self.task == "regression":
            out = Dense(self.num_outputs, activation='linear', name='output')(out)
            model = Model(inputs=[atoms, adjms, dists], outputs=out)
            model.compile(optimizer=Adam(lr=0.001), 
                          loss=self.loss, 
                          metrics=[std_mae(std=self.std), 
                                   std_rmse(std=self.std), 
                                   std_r2(std=self.std)])
        elif self.task == "binary":
            out = Dense(self.num_outputs, activation='sigmoid', name='output')(out)
            model = Model(inputs=[atoms, adjms, dists], outputs=out)
            model.compile(optimizer=Adam(lr=0.001), loss=self.loss)
        elif self.task == "classification":
            out = Dense(self.num_outputs, activation='softmax', name='output')(out)
            model = Model(inputs=[atoms, adjms, dists], outputs=out)
            model.compile(optimizer=Adam(lr=0.001), loss=self.loss)
        else:
            raise ValueError("Unsupported task on model generation.")

        return model