from typing import Any, Dict, List, Optional, Union

import pandas as pd

from profit.dataset.parsers.data_frame_parser import DataFrameParser
from profit.dataset.preprocessing.mutator import PDBMutator
from profit.dataset.preprocessors.base_preprocessor import BasePreprocessor


class JSONFileParser(DataFrameParser):
    """Javascript Object Notation (JSON) file parser.
    
    The .json file should contain a key which has the sequence as  
    input, and label key which is the target to predict.

    Params:
    -------
    preprocessor: BasePreprocessor
        Preprocessor instance.
    
    mutator: PDBMutator or None, optional, default=None
        Mutator instance. Checks if mutation type is compatible with  
        preprocessor instance.

    data_key: str, optional, default="Variants"
        Data key.

    pdb_key: str or None, optional, default=None
        Protein Data Bank (PDB) id key. Contains ID associated with 
        the data sequence. If data_key contains SMILES strings, then 
        this value is ignored. If data_key contains sequence, and 
        mutations are being performed, then the value contained in this 
        key is used to download/get the full structural information 
        associated with the sequence. 

    pos_key: str or None, optional, default=None
        Position key. Contains information about which residues to 
        modify. If None, along with mutator, no mutations are performed 
        at the specified positions. Useful for performing mutations 
        on-the-fly, without need for preprocessing mutations beforehand. 

    labels: str or list of str or None, optional, default=None
        Label key(s). If None, label keys are not extracted.

    process_as_seq: bool, optional, default=True
        If True, process data contained in data_col as a sequence.
        Otherwise, it is considered a SMILES string. 
    """

    def __init__(self, preprocessor: BasePreprocessor, 
                 mutator: Optional[PDBMutator]=None, 
                 data_key: str="Variants", 
                 pdb_key: Optional[str]=None, 
                 pos_key: Optional[str]=None,
                 labels: Optional[Union[str, List[str]]]=None, 
                 process_as_seq: bool=True) -> None:
        super(JSONFileParser, self).__init__(preprocessor, mutator, \
            data_col=data_key, pdb_col=pdb_key, pos_col=pos_key, labels=labels, 
            process_as_seq=process_as_seq)


    def parse(self, filepath: str, 
              target_index: Optional[List[int]]=None, 
              return_is_successful: bool=True) -> Dict[str, Any]:
        """Parse the .json file using the preprocessor given.

        Label is extracted from `labels` key and input features are 
        extracted from sequence information in `data_key` key.
                
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
        df = pd.read_json(filepath)
        return super(JSONFileParser, self).parse(df, target_index=target_index, \
            return_is_successful=return_is_successful)
