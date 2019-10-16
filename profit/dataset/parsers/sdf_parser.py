from typing import Any, Dict, List, Optional, Union

import numpy as np

from tqdm import tqdm
from rdkit.Chem import rdmolfiles

from profit.dataset.parsers.base_parser import BaseFileParser
from profit.dataset.preprocessors.base import BasePreprocessor
from profit.dataset.preprocessors.common import MolFeatureExtractionError


class SDFFileParser(BaseFileParser):
    """Structure data file (SDF) file parser.
    
    Params:
    -------
    preprocessor: BasePreprocessor
        Preprocessor instance.

    labels: str or list of str or None, optional, default=None
        Label column(s).
    """

    def __init__(self, preprocessor: BasePreprocessor, 
                 labels: Optional[Union[str, List[str]]]=None) -> None:
        super(SDFFileParser, self).__init__(preprocessor)
        self.labels = labels

    
    def parse(self, filepath: str, return_smiles: bool=False, 
              target_index: Optional[List[int]]=None) -> Dict[str, Any]:
        """Parse SDF file using the preprocessor given.
        
        Params:
        -------
        filepath: str
            Path to the dataset to be parsed.
        
        return_smiles: bool, optional, default=False
            Whether or not to return the SMILES string of the molecule. If 'True', this function 
            returns the processed dataset and the smiles string. If 'False', then None is returned 
            for the smiles string.

        target_index: list of int or None, optional, default=None
            Indicies of molecules to extract. If None, then all examples are parsed. Allows for  
            easier batching.

        Returns:
        --------
        out: dict

        """
        smiles_list = []
        pp = self.preprocessor

        if isinstance(pp, BasePreprocessor):
            # Read molecules from filepath
            mols = rdmolfiles.SDMolSupplier(filepath)
            if target_index is None:
                target_index = list(range(len(mols)))

            features = None
            fail_count = 0
            success_count = 0
            for idx in tqdm(target_index):
                mol = mols[idx]

                if mol is None:
                    fail_count += 1
                    continue

                try:
                    # Extract labels before standardization of smiles
                    if self.labels is not None:
                        labels = pp.get_labels(mol, self.labels)
                    
                    # Obtain canonical smiles (since equivalent smiles str are not unique)
                    canonical_smiles, _ = pp.prepare_mol(mol)
                    input_feats = pp.get_input_feats(mol)
                    
                    # Initialize features: list of list
                    if features is None:
                        num_feats = len(input_feats) if isinstance(input_feats, tuple) else 1
                        if self.labels is not None:
                            num_feats += 1
                        features = [[] for _ in range(num_feats)]

                    # Add canonical smiles, if required
                    if return_smiles:
                        smiles_list.append(canonical_smiles)
                except MolFeatureExtractionError as e:
                    # If the feature extraction of the mol fails, skip
                    fail_count += 1
                    continue
                except Exception as e:
                    # If for some other reason mol cannot be read/parsed properly, skip
                    print('Error while parsing mol {}, type: {}, {}'.format(
                        idx, type(e).__name__, e.args))
                    fail_count += 1
                    continue

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
            
            # Compile each feature(s) into individual np.ndarray s.t. the first channel becomes 
            # the num of examples in the dataset.
            all_feats = [np.array(feature) for feature in features]
            print('Preprocess finished. FAIL {}, SUCCESS {}, TOTAL {}'.format(
                fail_count, success_count, len(target_index)))
        else:
            raise NotImplementedError
        
        all_smiles = np.array(smiles_list) if return_smiles else None
        return {"dataset": all_feats, "smiles": all_smiles}


if __name__ == "__main__":
    from profit.dataset.preprocessors.gcn_preprocessor import GCNPreprocessor
    preprocessor = GCNPreprocessor(max_atoms=49, out_size=49)
    parser = SDFFileParser(preprocessor)
    result = parser.parse("data/processed/fitness570-constraints=1-algo=ETKDG.sdf", 
                          return_smiles=True, target_index=[0,2])
    dataset = result['dataset']
    smiles = result['smiles']
    print(len(dataset))
    print(dataset[0].shape, dataset[1].shape, dataset[2].shape)
    print(smiles)