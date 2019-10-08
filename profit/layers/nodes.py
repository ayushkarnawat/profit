"""Operations for computation of scalar and vector features.

Used before every layer's convolution, pooling, and activation step.
"""

import tensorflow as tf

from keras import initializers, regularizers, activations
from keras.layers import Layer

from typing import Any, Dict, List, Optional, Tuple, Union


class GraphSToS(Layer):
    """Scalar to scalar computation.
    
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
