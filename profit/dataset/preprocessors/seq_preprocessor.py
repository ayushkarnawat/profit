from typing import List, Optional, Union

import numpy as np
from profit.dataset.preprocessors.base import BasePreprocessor


class SequencePreprocessor(BasePreprocessor):
    """Preprocessor for sequences."""

    def __init__(self):
        super(SequencePreprocessor, self).__init__()


    def get_input_feats(self, seq: Union[str, List[str]]) -> np.ndarray:
        """Get the respective features for the sequence.

        NOTE: Each subclass must override this method.
        
        Params:
        -------
        seq: str or list of str
            The sequence, which contains human-readable representation of 
            amino acid names, whose featues will be computed.
        """
        raise NotImplementedError
    