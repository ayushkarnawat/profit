from typing import Any, Dict, List, Optional, Union

import pandas as pd

from profit.dataset.parsers.data_frame_parser import DataFrameParser
from profit.dataset.preprocessing.mutator import PDBMutator
from profit.dataset.preprocessors.base import BasePreprocessor


class CSVFileParser(DataFrameParser):
    """Comma-seperated values (CSV) parser.
    
    The .csv file should contain column which has the sequence as  
    input, and label column which is the target to predict.

    Params:
    -------
    preprocessor: BasePreprocessor
        Preprocessor instance.
    
    mutator: PDBMutator or None, optional, default=None
        Mutator instance. Used to check if mutation type is compatible 
        with preprocessor instance.

    data_col: str, optional, default="Variants"
        Data column.

    labels: str or list of str or None, optional, default=None
        Label column(s).
    """

    def __init__(self, preprocessor: BasePreprocessor, 
                 mutator: Optional[PDBMutator]=None, 
                 data_col: str="Variants", 
                 labels: Optional[Union[str, List[str]]]=None) -> None:
        super(CSVFileParser, self).__init__(preprocessor, mutator, 
                                            data_col=data_col, labels=labels)


    def parse(self, filepath: str, 
              target_index: Optional[List[int]]=None, 
              return_is_successful: bool=True) -> Dict[str, Any]:
        """Parse the .csv file using the preprocessor given.

        Label is extracted from `labels` columns and input features are 
        extracted from sequence information in `data_col` column.
        
        TODO: Add ability to mutate sequence at specified positions.
        
        Params:
        -------
        filepath: str
            The filepath to be parsed.

        target_index: list of int or None, optional, default=None
            Indicies to extract. If None, then all examples (in the 
            dataset) are parsed. Allows for easier batching.

        return_is_successful: bool, optional, default=True
            If True, boolean list (representing whether parsing of the 
            sequence has succeeded or not) is returned in the key 
            'is_successful'. If False, `None` is returned instead.
        """
        df = pd.read_csv(filepath, sep=',')
        return super(CSVFileParser, self).parse(df, target_index=target_index, 
                                                return_is_successful=return_is_successful)
