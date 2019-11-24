import traceback

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from tqdm import tqdm

from profit.dataset.parsers.base_parser import BaseFileParser
from profit.dataset.preprocessing.mutator import PDBMutator
from profit.dataset.preprocessing.seq_feats import SequenceFeatureExtractionError
from profit.dataset.preprocessors.base import BasePreprocessor
from profit.dataset.preprocessors.seq_preprocessor import SequencePreprocessor


class DataFrameParser(BaseFileParser):
    """DataFrame parser.
    
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
        super(DataFrameParser, self).__init__(preprocessor, mutator)
        self.data_col = data_col
        if isinstance(labels, str):
            labels = [labels]
        self.labels = labels


    def parse(self, df: pd.DataFrame, 
              target_index: Optional[List[int]]=None, 
              return_is_successful: bool=True) -> Dict[str, Any]:
        """Parse dataframe using the preprocessor given.

        TODO: Add ability to mutate sequence at specified positions.
        
        Params:
        -------
        df: pd.DataFrame
            DataFrame to be parsed.

        target_index: list of int or None, optional, default=None
            Indicies to extract. If None, then all examples (in the dataset) 
            are parsed. Allows for easier batching.

        return_is_successful: bool, optional, default=True
            If True, boolean list (representing whether parsing of the 
            sequence has succeeded or not) is returned in the key 
            'is_successful'. If False, `None` is returned instead.
        """
        pp = self.preprocessor
        mutator = self.mutator
        features = None
        is_successful_list = []

        if isinstance(pp, SequencePreprocessor):
            if target_index is not None:
                df = df.iloc[target_index]

            data_index = df.columns.get_loc(self.data_col)
            labels_index = [] if self.labels is None else [df.columns.get_loc(l) for l in self.labels]

            total_count = df.shape[0]
            fail_count = 0
            success_count = 0 
            for row in tqdm(df.itertuples(index=False), total=total_count):
                seq = row[data_index]
                labels = [row[i] for i in labels_index]

                # Check for valid data input (aka sequence)
                if seq is None:
                    fail_count += 1
                    if return_is_successful:
                        is_successful_list.append(False)
                    continue   

                # Obtain sequence features
                try:
                    input_feats = pp.get_input_feats(seq)
                except SequenceFeatureExtractionError as e:
                    # If feature extraction of sequence fails, skip
                    fail_count += 1
                    if return_is_successful:
                        is_successful_list.append(False)
                    continue
                except Exception as e:
                    # If for some other reason the sequence cannot be parsed properly, skip
                    print('Error while parsing `{}`, type: {}, {}'.format(seq, \
                        type(e).__name__, e.args))
                    traceback.print_exc()
                    fail_count += 1
                    if return_is_successful:
                        is_successful_list.append(False)
                    continue

                # Initialize features: list of lists
                if features is None:
                    num_feats = len(input_feats) if isinstance(input_feats, tuple) else 1
                    if self.labels is not None:
                        num_feats += 1
                    features = [[] for _ in range(num_feats)]
                
                # Append computed features to respective cols
                if isinstance(input_feats, tuple):
                    for i in range(len(input_feats)):
                        features[i].append(input_feats[i])
                else:
                    features[0].append(input_feats)
                
                # Add label values as last column, if provided
                if self.labels is not None:
                    features[len(features) - 1].append(labels)
                
                success_count += 1
                if return_is_successful:
                    is_successful_list.append(True)

            print('Preprocess finished. FAIL {}, SUCCESS {}, TOTAL {}'.format(\
                fail_count, success_count, total_count))
        else:
            raise NotImplementedError

        # Compile each feature(s) into individual np.ndarray s.t. the first 
        # channel becomes the num of examples in the dataset.
        all_feats = [np.array(feature) for feature in features]
        is_successful = np.array(is_successful_list) if return_is_successful else None
        return {"dataset": all_feats, "is_successful": is_successful}
