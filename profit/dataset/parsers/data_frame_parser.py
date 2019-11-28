import traceback
import warnings

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from rdkit.Chem import rdmolfiles
from tqdm import tqdm

from profit.dataset.parsers.base_parser import BaseFileParser
from profit.dataset.preprocessing.mutator import PDBMutator
from profit.dataset.preprocessors import preprocess_method_dict
from profit.dataset.preprocessors.base_preprocessor import BasePreprocessor
from profit.dataset.preprocessors.seq_preprocessor import SequencePreprocessor
from profit.dataset.preprocessors.mol_preprocessor import MolPreprocessor


class DataFrameParser(BaseFileParser):
    """DataFrame parser.
    
    Params:
    -------
    preprocessor: BasePreprocessor
        Preprocessor instance.
    
    mutator: PDBMutator or None, optional, default=None
        Mutator instance. Checks if mutation type is compatible with  
        preprocessor instance.

    data_col: str, optional, default="Variants"
        Data column.

    pdb_col: str or None, optional, default=None
        Protein Data Bank (PDB) id column. Contains ID associated with 
        the data sequence. If data_col contains SMILES strings, then 
        this value is ignored. If data_col contains sequence, and 
        mutations are being performed, then the value contained in this 
        column is used to download/get the full structural information 
        associated with the sequence. 

    pos_col: str or None, optional, default=None
        Position column. Contains information about which residues to 
        modify. If None, along with mutator, no mutations are performed 
        at the specified positions. Useful for performing mutations 
        on-the-fly, without need for preprocessing mutations beforehand. 

    labels: str or list of str or None, optional, default=None
        Label column(s). If None, label columns are not extracted.

    process_as_seq: bool, optional, default=True
        If True, process data contained in data_col as a sequence.
        Otherwise, it is considered a SMILES string.
    """

    def __init__(self, preprocessor: BasePreprocessor, 
                 mutator: Optional[PDBMutator]=None, 
                 data_col: str="Variants", 
                 pdb_col: Optional[str]=None, 
                 pos_col: Optional[str]=None, 
                 labels: Optional[Union[str, List[str]]]=None, 
                 process_as_seq: bool=True) -> None:
        super(DataFrameParser, self).__init__(preprocessor, mutator)
        self.data_col = data_col
        self.pdb_col = pdb_col
        self.pos_col = pos_col
        if isinstance(labels, str):
            labels = [labels]
        self.labels = labels
        self.process_as_seq = process_as_seq


    def parse(self, df: pd.DataFrame, 
              target_index: Optional[List[int]]=None, 
              return_is_successful: bool=True) -> Dict[str, Any]:
        """Parse dataframe using the preprocessor given.

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
        features = None
        is_successful_list = []
        pp = self.preprocessor
        mutator = self.mutator
        processed_as = 'sequence' if self.process_as_seq else 'SMILES'

        if target_index is not None:
            df = df.iloc[target_index]

        data_index = df.columns.get_loc(self.data_col)
        pdb_index = df.columns.get_loc(self.pdb_col) if self.pdb_col is not None else None
        pos_index = df.columns.get_loc(self.pos_col) if self.pos_col is not None else None
        labels_index = [] if self.labels is None else [df.columns.get_loc(l) for l in self.labels]

        fail_count = 0
        success_count = 0 
        total_count = df.shape[0]
        for row in tqdm(df.itertuples(index=False), total=total_count):
            data: Optional[Union[str, List[str]]] = row[data_index]
            pdbid = row[pdb_index] if pdb_index is not None else None
            positions = row[pos_index] if pos_index is not None else None
            labels = [row[i] for i in labels_index]

            try:
                # Check for valid data input
                if data is None:
                    raise TypeError("Invalid type: {}. Should be str or list " \
                        "of str.".format(type(data).__name__))
                elif len(data) == 0:
                    # Raise error for now, if empty list or str is passed in. 
                    # TODO: Change how each type (molecule or sequence) feature 
                    # processing handles empty data. If mol.GetNumAtoms() == 0 
                    # or len(seq) == 0, then a respective FeatureExtractionError 
                    # should be raised. 
                    raise ValueError("Cannot process empty data.")
                
                # SMILES parsing
                if not self.process_as_seq:
                    if mutator is not None:
                        warnings.warn("SMILES string '{}' cannot be mutated.".format(data))

                    # SMILES string can only be processed as rdkit.Mol instance.
                    mol = rdmolfiles.MolFromSmiles(data, sanitize=True)
                    if mol is None:
                        raise TypeError("Invalid type: {}. Should be " \
                            "rdkit.Chem.rdchem.Mol.".format(type(mol).__name__))

                    # Compute features if its a proper molecule
                    if isinstance(pp, MolPreprocessor):
                        input_feats = pp.get_input_feats(mol)
                    else:
                        valid_preprocessors = [pp.__name__ for pp in preprocess_method_dict.values() 
                                               if isinstance(pp(), MolPreprocessor)]
                        raise ValueError("{} cannot compute features for SMILES-based input " \
                            "'{}'. Choose a valid SMILES-based preprocessor: {}.".format( \
                            type(pp).__name__, data, valid_preprocessors))
                else:
                    # Sequence-based parsing
                    if mutator is not None:
                        if pdbid is None:
                            raise ValueError("PDB ID not specified. Unable to mutate residue.")

                        if positions is None:
                            raise ValueError("Positions not specified. PDBMutator needs " \
                                "residue positions to mutate residues at defined locations.")
                        else:
                            # Raise error for now, as lengths of positions and seqs need to match 
                            # to work with the current implementation of mutator. 
                            # TODO: Change when implementation of mutator changes.
                            # NOTE: Should we assume that if the len(positions) < len(data), then 
                            # the user wants to modify those positions in the sequence?
                            if len(data) != len(positions):
                                raise ValueError("Length of input (N={}) is not the same as number " \
                                    "of positions (N={}) to modify. Did you pass in the full" \
                                    "sequence? Currently, mutations can only be performed with " \
                                    "information about which residue position(s) to modify and the " \
                                    "replacement residue(s) at those positions. If you want to " \
                                    "process only the input sequence (without any mutations), " \
                                    "set mutator=None.".format(len(data), len(positions)))
                            # Mutate residues (to primary or tertiary) based off mutator instance
                            replace_with = {resid: data[i] for i, resid in enumerate(positions)}
                            data = mutator.modify_residues(pdbid, replace_with=replace_with)

                        # Obtain features based on which preprocessor is used
                        if isinstance(pp, tuple(preprocess_method_dict.values())):
                            input_feats = pp.get_input_feats(data)
                        else:
                            raise NotImplementedError
                    else:
                        # Since it is not mutated, the data can now ONLY be a sequence
                        # (since 3D representation cannot be within a single column in a df)
                        if isinstance(pp, SequencePreprocessor):
                            input_feats = pp.get_input_feats(data)
                        else:
                            valid_preprocessors = [pp.__name__ for pp in preprocess_method_dict.values() 
                                                   if isinstance(pp(), SequencePreprocessor)]
                            raise ValueError("{} cannot compute features for sequence-based input " \
                                "'{}'. Either mutate data (by passing in PDBMutator instance) to " \
                                "'tertiary' structure or choose a valid sequence-based preprocessor: " \
                                "{}.".format(type(pp).__name__, data, valid_preprocessors))
            except Exception as e:
                # If for some reason the data cannot be parsed properly, skip
                print('Error while parsing `{}` as {}, type: {}, {}'.format(\
                    data, processed_as, type(e).__name__, e.args))
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

        # Compile each feature(s) into individual np.ndarray s.t. the first 
        # channel becomes the num of examples in the dataset.
        all_feats = [np.array(feature) for feature in features] if features is not None else []
        is_successful = np.array(is_successful_list) if return_is_successful else None
        return {"dataset": all_feats, "is_successful": is_successful}
