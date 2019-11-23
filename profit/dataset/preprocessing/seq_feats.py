import numpy as np

from typing import Any, List, Optional, Union
from profit.dataset.preprocessing.vocab import AA_DICT


class SequenceFeatureExtractionError(Exception): 
    pass


def check_num_residues(seq: Union[str, List[str]], 
                       max_num_residues: Optional[int]=-1) -> None:
    """Check if number of residues in sequence does not exceed `max_num_residues`.

    If number of residues in `seq` exceeds the number `max_num_residues`, 
    it will raise `SequenceFeatureExtractionError` exception.
    
    Params:
    -------
    seq: str or list of str
        The sequence, which contains human-readable representation of 
        amino acid names, to check.
        
    num_max_residues: int, optional , default=-1 
        Maximum allowed number of residues in a sequence/protein. If 
        negative, check passes unconditionally.
    """
    num_residues = len(seq)
    if max_num_residues >= 0 and num_residues > max_num_residues:
        raise SequenceFeatureExtractionError("Residues in sequence (N={}) exceeds " \
            "num_max_residues (N={}).".format(num_residues, max_num_residues))


def construct_embedding(seq: Union[str, List[str]], 
                        out_size: Optional[int]=-1, 
                        use_pretrained: Optional[bool]=False) -> np.ndarray:
    """Return the embedding of all amino acid residues in the string.  

    Params:
    -------
    seq: str or list of str
        The sequence, which contains human-readable representation of 
        amino acid names, to compute features for.
    
    out_size: int, optional, default=-1
        The size of the returned array. If this option is negative, it 
        does not take any effect. Otherwise, it must be larger than or 
        equal to the number of residues in the input molecule. If so, 
        the end of the array is padded with negative one.

    use_pretrained: bool, optional, default=False
        If 'True', then pre-trained amino acid embeddings are used. If 
        'False', the amino acid residues are only converted to ints 
        based on a vocab dictionary.

    Returns:
    --------
    embedding: np.ndarray
        Amino acid embedding of the protein. If `out_size` is non-
        negative, the returned matrix is equal to that value. Otherwise, 
        it is equal to the number of residues in the the sequence.

    Examples:
    ---------
    >>> seq1 = 'MTYKLILNGK'
    >>> construct_embedding(seq1, out_size=-1)
    [10. 16. 19.  8.  9.  7.  9. 11.  5.  8.]

    >>> seq2 = list(seq1)
    >>> construct_embedding(seq2, out_size=15)
    [10. 16. 19.  8.  9.  7.  9. 11.  5.  8. -1. -1. -1. -1. -1.]
    """
    num_residues = len(seq)

    # Determine output size to generate embedding matrix of same size for all sequences
    if out_size < 0:
        size = num_residues
    elif out_size >= num_residues:
        size = out_size
    else:
        raise ValueError("`out_size` (N={}) must be negative or larger than or "
                         "equal to the number of residues in the input sequence " 
                         "(N={}).".format(out_size, num_residues))

    # Convert residue names to int (based off vocab dict)
    if isinstance(seq, str):
        seq = list(seq)
    embedding = np.array([AA_DICT.get(aa) for aa in seq], dtype=np.float)

    # TODO: Embed into protein space (using preprocessing/embedding.py)
    # Embedding shape=(num_residues, embedding_dims)
    if use_pretrained:
        raise NotImplementedError
    
    # Pad (w/ unknown value) to defined size
    full_size = [size] + list(embedding.shape[1:])
    pad_width = [(0, full_size[i] - embedding.shape[i]) for i in range(embedding.ndim)]
    padded_embed = np.pad(embedding, pad_width, mode="constant", constant_values=AA_DICT.get("X"))
    return padded_embed
    