import tensorflow as tf
from keras.layers import Layer

from typing import List, Tuple, Union


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
        """Generate scalar and vector features for atoms and their distances, respectively. 

        Params:
        -------
        inputs: tf.Tensor or list/tuple of tf.Tensors.
            Inputs to the current layer.

        mask: tf.Tensor, list/tuple of tf.Tensors, or None, optional, default=None
            Previous layer(s) that feed into this layer, if available. Not used.

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

        For the embedding layer, since we have both scalar and vector features, we will have the 
        following shapes: (a) sfeats.shape=(k,N,M), and (b) vfeats.shape=(k,N,N,3). Here, `N` is 
        the number of atoms, `M` is the number of atom-level features in a molecule, and `k` is 
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
