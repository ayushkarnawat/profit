"""Common sequence-based features."""

from typing import List, Union

import numpy as np
from profit.utils.data_utils.tokenizers import AminoAcidTokenizer


class SequenceFeatureExtractionError(Exception):
    pass


def check_num_residues(seq: Union[str, List[str]],
                       max_num_residues: int = -1) -> None:
    """Check if number of residues in sequence does not exceed
    `max_num_residues`.

    If number of residues in `seq` exceeds the number `max_num_residues`,
    it will raise `SequenceFeatureExtractionError` exception.

    Params:
    -------
    seq: str or list of str
        The sequence, which contains human-readable representation of
        amino acid names, to check.

    num_max_residues: int , default=-1
        Maximum allowed number of residues in a sequence/protein. If
        negative, check passes unconditionally.
    """
    num_residues = len(seq)
    if max_num_residues >= 0 and num_residues > max_num_residues:
        raise SequenceFeatureExtractionError("Residues in sequence (N={}) exceeds " \
            "num_max_residues (N={}).".format(num_residues, max_num_residues))


def construct_embedding(seq: Union[str, List[str]],
                        vocab: str,
                        out_size: int = -1,
                        use_pretrained: bool = False) -> np.ndarray:
    """Return the embedding of all amino acid residues in the string.

    Params:
    -------
    seq: str or list of str
        The sequence, which contains human-readable representation of
        amino acid names, to compute features for.

    vocab: str
        Vocab dictionary used to convert sequence (i.e. amino acid
        residues) to ints via :class:`AminoAcidTokenizer`.

    out_size: int, default=-1
        The size of the returned array. If this option is negative, it
        does not take any effect. Otherwise, it must be larger than or
        equal to the number of residues in the input molecule. If so,
        the end of the array is padded with negative one.

    use_pretrained: bool, default=False
        If 'True', then pre-trained amino acid embeddings are used. If
        'False', the amino acid residues are only converted to ints
        based on a vocab dictionary (defined by vocab).

    Returns:
    --------
    embedding: np.ndarray
        Amino acid embedding of the protein. If `out_size` is non-
        negative, the returned matrix is equal to that value. Otherwise,
        it is equal to the number of residues in the the sequence.

    Examples:
    ---------
    >>> seq1 = 'MTYKLILNGK'
    >>> construct_embedding(seq1, vocab="iupac1", out_size=-1)
    [17 24 29 15 16 13 16 18 11 15]

    >>> seq2 = list(seq1)
    >>> construct_embedding(seq2, vocab="iupac1", out_size=15)
    [17 24 29 15 16 13 16 18 11 15  0  0  0  0  0]
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
    tokenizer = AminoAcidTokenizer(vocab)
    embedding = tokenizer.encode(seq)

    # TODO: Embed into protein space (using preprocessing/embedding.py)
    # Embedding shape=(num_residues, embedding_dims)
    if use_pretrained:
        raise NotImplementedError

    # Pad (w/ zero) to defined size
    try:
        full_size = [size] + list(embedding[:].shape[1:])
        pad_width = [(0, full_size[i] - embedding[:].shape[i]) for i in range(embedding.ndim)]
        padded_embed = np.pad(embedding, pad_width, mode="constant", \
            constant_values=tokenizer.convert_token_to_id(tokenizer.pad_token))
    except RuntimeError as e:
        # If there is no padding token
        if out_size > 0:
            print(f"Cannot pad to out_size={out_size} because {str(e)}")
        padded_embed = embedding
    return padded_embed
