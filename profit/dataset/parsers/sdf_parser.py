import traceback

from typing import Any, Dict, List, Optional, Union

import numpy as np

from tqdm import tqdm
from rdkit.Chem import rdmolfiles

from profit import backend as P
from profit.dataset.parsers.base_parser import BaseFileParser
from profit.dataset.preprocessing.mutator import PDBMutator
from profit.dataset.preprocessors.base_preprocessor import BasePreprocessor
from profit.dataset.preprocessors.mol_preprocessor import MolPreprocessor
from profit.utils.data_utils.cast import broadcast_array


class SDFFileParser(BaseFileParser):
    """Structure data file (SDF) file parser.
    
    Params:
    -------
    preprocessor: BasePreprocessor
        Preprocessor instance.

    mutator: PDBMutator or None, optional, default=None
        Mutator instance. Checks if mutation type is compatible with  
        preprocessor instance. Ignored for now.

    labels: str or list of str or None, optional, default=None
        Label column(s).
    """

    def __init__(self, preprocessor: BasePreprocessor, 
                 mutator: Optional[PDBMutator]=None, 
                 labels: Optional[Union[str, List[str]]]=None) -> None:
        super(SDFFileParser, self).__init__(preprocessor, mutator=None)
        if isinstance(labels, str):
            labels = [labels]
        self.labels = labels

    
    def parse(self, filepath: str,  
              target_index: Optional[List[int]]=None, 
              return_smiles: bool=True, 
              return_is_successful: bool=True) -> Dict[str, Any]:
        """Parse SDF file using the preprocessor given.

        TODO: Add ability to mutate sequence at specified positions.
        
        Params:
        -------
        filepath: str
            Path to the dataset to be parsed.

        target_index: list of int or None, optional, default=None
            Indicies to extract. If None, then all examples (in the 
            dataset) are parsed. Allows for easier batching.

        return_smiles: bool, optional, default=True
            Whether or not to return the SMILES string of the molecule. 
            If 'True', list of SMILES string is returned in the key 
            'smiles'. If 'False', `None` is returned instead.

        return_is_successful: bool, optional, default=True
            If True, boolean list (representing whether parsing of the 
            sequence has succeeded or not) is returned in the key 
            'is_successful'. If False, `None` is returned instead.

        Returns:
        --------
        out: dict
            Extracted features (aka dataset) and associated smiles of requested mol.  
        """
        smiles_list = []
        is_successful_list = []
        pp = self.preprocessor

        if isinstance(pp, MolPreprocessor):
            # Read molecules from filepath
            mols = rdmolfiles.SDMolSupplier(filepath)
            if target_index is None:
                target_index = list(range(len(mols)))

            features = None
            fail_count = 0
            success_count = 0
            total_count = len(target_index)
            for idx in tqdm(target_index, total=total_count):
                try:
                    # Check if rdkit.Mol instance
                    mol = mols[idx]
                    if mol is None:
                        raise TypeError("Invalid type: {}. Should be " \
                            "rdkit.Chem.rdchem.Mol.".format(type(mol)))

                    # Extract labels before standardization of smiles
                    if self.labels is not None:
                        labels = pp.get_labels(mol, self.labels)
                    
                    # Obtain canonical smiles (since equivalent smiles str are not unique)
                    canonical_smiles, _ = pp.prepare_mol(mol)
                    input_feats = pp.get_input_feats(mol)

                    # Add canonical smiles, if required
                    if return_smiles:
                        smiles_list.append(canonical_smiles)
                except Exception as e:
                    # If for some reason the data cannot be parsed properly, skip
                    print('Error while parsing `{}`, type: {}, {}'.format(mol, \
                        type(e).__name__, e.args))
                    traceback.print_exc()
                    fail_count += 1
                    if return_is_successful:
                        is_successful_list.append(False)
                    continue

                # Initialize features: list of list
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
        else:
            raise NotImplementedError
        
        print('Preprocess finished. FAIL {}, SUCCESS {}, TOTAL {}'.format(\
            fail_count, success_count, total_count))
        
        # Compile feature(s) into individual np.ndarray(s), padding each to max  
        # dims, if necessary. NOTE: The num of examples in the dataset depends 
        # on the data_format specified (represented by first/last channel).
        all_feats = [broadcast_array(feature) for feature in features] if features else []
        if P.data_format() == "batch_last":
            all_feats = [np.moveaxis(feat, 0, -1) for feat in all_feats]
        all_smiles = np.array(smiles_list) if return_smiles else None
        is_successful = np.array(is_successful_list) if return_is_successful else None
        return {"dataset": all_feats, "smiles": all_smiles, "is_successful": is_successful}