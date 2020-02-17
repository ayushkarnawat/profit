import tensorflow as tf

from keras import activations
from keras.layers import Layer

from typing import Any, Dict, List, Optional, Tuple, Union


class GraphGather(Layer):
    """Combines both scalar and vector features into similar representations. Essentially, like a 
    concatination layer.
    
    Params:
    -------
    pooling: str, optional, default="sum"
        Type of down-sample pooling technique to perform on the hidden representations after the 
        graph convolutions. The representations are either summed, averaged, or maxed depending 
        on the pooling type chosen.

    system: str, optional, default="cartesian"
        Whether to represent the data in spherical :math:`(r, \\theta, \\phi)` or cartesian (XYZ) 
        coordinates. 

    activation: str or None, optional, default=None
        Which activation to apply to the hidden representations after pooling. See 
        keras.activations for list of all possible activation functions. If 'None', then no 
        activation is performed (aka linear activation), and the hidden representation is simply 
        returned. This is not recommened as non-linear activations model tasks better.
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
        """Create the layer's weight with the respective params defined above. 
        
        Params:
        -------
        input_shape: tf.Tensor or list/tuple of tf.Tensors
            Input tensor(s) to reference for weight shape computations.
        """
        super(GraphGather, self).build(inputs_shape)


    def call(self, inputs: Union[tf.Tensor, Union[List[tf.Tensor], Tuple[tf.Tensor, ...]]], 
             mask: Union[tf.Tensor, Union[List[tf.Tensor], Tuple[tf.Tensor, ...]]]=None) \
             -> tf.Tensor:
        """Gather and concatenate both scalar and vector features.
        
        More specifically, we (a) downsample both features through pooling, (b) apply a non-linear 
        activation, and (c) convert to spherical coordinates, if needed.
        
        Params:
        -------
        inputs: tf.Tensor or list/tuple of tf.Tensors.
            Inputs to the current layer.

        mask: tf.Tensor, list/tuple of tf.Tensors, or None, optional, default=None
            Previous layer(s) that feed into this layer, if available. Not used.

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

        For the gather layer, since we have both scalar and vector features, we will have the 
        following shapes: (a) sfeats.shape=(k,M), and (b) vfeats.shape=(k,3,M). Here, `M` is the 
        number of atom-level features in a molecule, and `k` is the number of samples in the 
        batch/dataset.

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
