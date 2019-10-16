from typing import Any, Dict
from profit.dataset.preprocessors.base import BasePreprocessor


class BaseFileParser(object):
    """Base class for file parsing.
    
    Params:
    -------
    preprocessor: BasePreprocessor
        Preprocessor instance.
    """

    def __init__(self, preprocessor: BasePreprocessor) -> None:
        self.preprocessor = preprocessor


    def parse(self, filepath: str) -> Dict[str, Any]:
        """Parse the given file. Must be overridden by each subclass.
        
        Params:
        -------
        filepath: str
            Path to the dataset to be parsed.
        """
        raise NotImplementedError
    