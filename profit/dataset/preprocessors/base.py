from typing import List, Optional, Tuple
from rdkit import Chem

class BasePreprocessor(object):
    """Preprocessor for rdkit.Chem.Mol instance. 
    
    Params:
    -------
    add_Hs: bool, optional, default=False
        Whether or not to add H's to the molecule.

    kekulize: bool, optional, default=False
        Whether or not to kekulize the molecule. If True, use the Kekule form (no aromatic bonds) 
        in the SMILES rep.
    """

    def __init__(self, add_Hs: Optional[bool]=False, kekulize: Optional[bool]=False) -> None:
        self.add_Hs = add_Hs
        self.kekulize = kekulize

    
    def prepare_mol(self, mol: Chem.Mol) -> Tuple[str, Chem.Mol]:
        """Prepare both smiles and mol by standardizing to common rules.

        This method should be called before `get_input_feats`.

        Params:
        -------
        mol: rdkit.Chem.Mol
            Molecule of interest.

        Returns:
        --------
        canonical_smiles: str
            Canonical SMILES representation of the molecule. 

        mol: rdkit.Chem.Mol
            Modified molecule w/ kekulization and Hs added, if specified.
        """
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        mol = Chem.MolFromSmiles(canonical_smiles)

        if self.add_Hs:
            mol = Chem.AddHs(mol)
        if self.kekulize:
            Chem.Kekulize(mol)
        return canonical_smiles, mol

    
    def get_labels(self, mol: Chem.Mol, label_names: Optional[List[str]]=None) -> List[str]:
        """Extract corresponding label info from the molecule.
        
        Params:
        -------
        mol: rdkit.Chem.Mol
            Molecule of interest.

        label_names: list of str or None, optional, default=None
            Name of labels.

        Returns:
        --------
        labels: list of str
            Label info, its length is equal to that of `label_name`.
        """
        if label_names is None:
            return []
        return [mol.GetProp(name) if mol.HasProp(name) else None for name in label_names]
    

    def get_input_feats(self, mol: Chem.Mol):
        """Get the respective features for the molecule. 

        NOTE: Each subclass must override this method.

        Params:
        -------
        mol: rdkit.Chem.Mol
            Molecule of interest whose features will be extracted.
        """
        raise NotImplementedError
