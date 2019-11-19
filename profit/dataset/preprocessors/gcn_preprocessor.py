import numpy as np

from rdkit.Chem import rdchem
from typing import Optional, Tuple

from profit.dataset.preprocessing.mol_feats import check_num_atoms
from profit.dataset.preprocessing.mol_feats import construct_adj_matrix
from profit.dataset.preprocessing.mol_feats import construct_mol_features
from profit.dataset.preprocessing.mol_feats import construct_pos_matrix
from profit.dataset.preprocessors.mol_preprocessor import MolPreprocessor


class GCNPreprocessor(MolPreprocessor):
    """Graph Convolution Neural Network (GCN) processor.
    
    Params:
    -------
    max_atoms: int, optional, default=-1
        Maximum allowed number of atoms in a molecule. If negative, there is no limit.

    out_size: int, optional, default=-1
        The size of the returned array (used by `get_input_feats`). If this option is negative, 
        it does not take any effect. Otherwise, it must be larger than or equal to the number of 
        atoms in the input molecule. If so, the end of the array is padded with zeros.

    add_Hs: bool, optional, default=False
        Whether or not to add H's to the molecule.

    kekulize: bool, optional, default=True
        Whether or not to kekulize the molecule. If True, use the Kekule form (no aromatic bonds) 
        in the SMILES rep.
    """
    
    def __init__(self, max_atoms: Optional[int]=-1, out_size: Optional[int]=-1, 
                 add_Hs: Optional[bool]=False, kekulize: Optional[bool]=True):
        super(GCNPreprocessor, self).__init__(add_Hs=add_Hs, kekulize=kekulize)

        if max_atoms >= 0 and out_size >= 0 and max_atoms > out_size:
            raise ValueError('max_atoms (N={0:d}) must be less than or equal to out_size'
                             '(N={1:d})'.format(max_atoms, out_size))
        self.max_atoms = max_atoms
        self.out_size = out_size

    
    def get_input_feats(self, mol: rdchem.Mol) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute input features for the molecule.
        
        Params:
        -------
        mol: rdkit.Chem.rdchem.Mol
            Molecule of interest whose features will be extracted.

        Returns:
        --------
        mol_feats: np.ndarray, shape=(n,m)
            Molecule features, where `n` is the total number of atoms in the mol, and `m` is the 
            number of feats.

        adj_matrix: np.ndarray, shape=(n,n)
            Adjacency matrix of input molecule. 

        pos_matrix: np.ndarray, shape=(n,n,3)
            Relative position matrix. 
        """
        check_num_atoms(mol, self.max_atoms)
        mol_feats = construct_mol_features(mol, out_size=self.out_size)
        adj_matrix = construct_adj_matrix(mol, out_size=self.out_size, normalize=True)
        pos_matrix = construct_pos_matrix(mol, out_size=self.out_size)
        return mol_feats, adj_matrix, pos_matrix
