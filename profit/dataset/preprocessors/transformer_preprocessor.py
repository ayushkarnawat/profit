from typing import List, Optional, Union

import numpy as np
from profit.dataset.preprocessing.seq_feats import check_num_residues
from profit.dataset.preprocessing.seq_feats import construct_embedding
from profit.dataset.preprocessors.seq_preprocessor import SequencePreprocessor


class TransformerPreprocessor(SequencePreprocessor):
    """Transformer preprocessor.
    
    Params:
    -------
    max_residues: int, optional, default=-1
        Maximum allowed number of residues in a molecule. If negative, 
        there is no limit.

    out_size: int, optional, default=-1
        The size of the returned array (used by `get_input_feats`). If 
        this option is negative, it does not take any effect. Otherwise, 
        it must be larger than or equal to the number of residues in the 
        input molecule. If so, the end of the array is padded with zeros.

    use_pretrained: bool, optional, default=False
        If 'True', then pre-trained amino acid embeddings are used. If 
        'False', the amino acid residues are only converted to ints 
        based on a vocab dictionary.
    """

    def __init__(self, max_residues: Optional[int]=-1, 
                 out_size: Optional[int]=-1, 
                 use_pretrained: bool=False) -> None:
        super(TransformerPreprocessor, self).__init__()

        if max_residues >= 0 and out_size >= 0 and max_residues > out_size:
            raise ValueError('max_residues (N={0:d}) must be less than or equal ' \
                'to out_size (N={1:d}).'.format(max_residues, out_size))
        self.max_residues = max_residues
        self.out_size = out_size
        self.use_pretrained = use_pretrained

    
    def get_input_feats(self, seq: Union[List[str], str]) -> np.ndarray:
        """Compute input features for the sequence.
        
        Params:
        -------
        seq: str or list of str
            The sequence, which contains human-readable representation of 
            amino acid names, whose featues will be computed.

        Returns:
        --------
        embedding: np.ndarray
            Amino acid embedding of the protein.
        """
        check_num_residues(seq, self.max_residues)
        embedding = construct_embedding(seq, self.out_size, self.use_pretrained)
        return embedding
        