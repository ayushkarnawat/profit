from typing import Any, Dict, Optional

from profit.dataset.preprocessing.mutator import PDBMutator
from profit.dataset.preprocessors.base_preprocessor import BasePreprocessor
from profit.dataset.preprocessors.mol_preprocessor import MolPreprocessor
from profit.dataset.preprocessors.seq_preprocessor import SequencePreprocessor


class BaseFileParser(object):
    """Base class for file parsing.
    
    Params:
    -------
    preprocessor: BasePreprocessor
        Preprocessor instance.

    mutator: PDBMutator or None, optional, default=None
        Mutator instance. Used to check if mutation type is compatible 
        with preprocessor instance.
    """

    def __init__(self, preprocessor: BasePreprocessor, 
                 mutator: Optional[PDBMutator]=None) -> None:
        # Check if mutation type is compatible with preprocessor
        if mutator is not None:
            if isinstance(preprocessor, SequencePreprocessor) and mutator.fmt != "primary":
                raise ValueError("{} only works with protein primary structure. " \
                    "Cannot mutate to {} structure.".format(type(preprocessor).__name__, mutator.fmt))
            elif isinstance(preprocessor, MolPreprocessor) and mutator.fmt != "tertiary":
                raise ValueError("{} only works with protein tertiary structure. " \
                    "Cannot mutate to {} structure.".format(type(preprocessor).__name__, mutator.fmt))
        self.mutator = mutator
        self.preprocessor = preprocessor


    def parse(self, filepath: str) -> Dict[str, Any]:
        """Parse the given file. Must be overridden by each subclass.
        
        Params:
        -------
        filepath: str
            Path to the dataset to be parsed.
        """
        raise NotImplementedError
    