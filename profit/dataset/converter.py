"""Module to convert raw dataset into processed form."""

import os
import logging
import multiprocessing as mp

from typing import Tuple, Optional, Any


import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

from profit.utils import load_csv
from profit.cyclops import struct_gen as sg


# Setup logging
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s  %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def convert_to_smiles(filepath: str, 
                      save_path: str, 
                      x_name: str, 
                      y_name: str, 
                      save_constraints: Optional[bool]=False) -> pd.DataFrame:
    """
    Convert the data to an intermediate format which include the peptide's SMILES string.

    Params:
    -------
    filepath: str
        Path where the dataset is located.

    save_path: str
        Path to store modified data.

    x_name: str
        Column name of data X.

    y_name: str
        Column name of labels associated with the data X.

    save_constraints: bool, optional, default=False
        Whether to save the chemical SMILES with constraints if available.

    Returns:
    --------
    smiles_df: pd.DataFrame
        Converted peptide string to SMILES with specified constraints.
    """
    # Load data
    logger.info('Loading dataset from `{0:s}`...'.format(filepath))
    X,y = load_csv(filepath, x_name=x_name, y_name=y_name)

    # Setup pandas df
    cols = ['Variant', 'Constraints', 'SMILES']
    smiles_df = pd.DataFrame(data=None, columns=cols)

    logger.info('Converting peptides to SMILES strings')
    structs = sg.gen_structs_from_seqs(X, ssbond=True, htbond=True, scctbond=True, 
                                       scntbond=True, scscbond=True, linear=True)
    
    for idx, struct in enumerate(structs):
        smiles_df.loc[idx] = list(struct)
    
    # Exploit the way the constraints are saved in the df. If there are constraints for a peptide, 
    # they are stored first, followed by the linear one (unconstrained). NOTE: This could break if 
    # the constraints are saved in a different fashion (i.e. last instead of first). 
    smiles_df = smiles_df.drop_duplicates('Variant', keep='first') if save_constraints \
        else smiles_df.drop_duplicates('Variant', keep='last')
    
    # Since there is only one instance of each variant, we can simply store the scores. NOTE: 
    # Ideally should check if each "fitness" score is associated with the right variant.  
    smiles_df['Fitness'] = y
    
    # Save dataset to file. Checks if intended filepath is available.
    save_path = os.path.expanduser(save_path)
    save_dir, _ = os.path.split(save_path)
    if not os.path.exists(save_dir):
        logger.info('Creating directory `{0:s}`'.format(save_dir))
        os.makedirs(save_dir)
    smiles_df.to_csv(save_path, index=False)
    logger.info('Saved dataset to `{0:s}`'.format(save_path))
    return smiles_df


def optimize_coords(idx: int, 
                    m: Chem.Mol, 
                    prop: Any, 
                    algo: Optional[str]="ETKDG") -> Tuple[Chem.Mol, Any]:
    """
    Optimize 3D coordinates for each compound. Defines the XYZ positions for each atom in the 
    molecule. 

    In order to get accurate 3D conformations, it is a good idea to add H's to molecule first, 
    embed (write) and optimize the coordinates, and then remove the unwanted H's. 

    NOTE: Conformation generation is a difficult and subtle task. The original 2D->3D conversion
    provided with the RDKit was not intended to be a replacement for a “real” conformational 
    analysis tool; it merely provides quick 3D structures for cases when they are required. We 
    believe, however, that the newer ETKDG method[#riniker2]_ should be adequate for most purposes.
    See: https://www.rdkit.org/docs/GettingStartedInPython.html#working-with-3d-molecules.

    Additionally, using the 'UFF' and 'MMFF' algorithm will generate multiple conformations for 
    each compound, optimize them using their respective force field, and then choose the 
    conformation that has the lowest energy. This is computationally very expensive.  

    Params:
    -------
    idx: int
        Index location of molecule within dataset.
    
    m: Chem.Mol
        The molecule to optimize.
    
    prop: Any
        Property associated with the molecule.

    algo: str, optional, default='ETKDG'
        Which force field algorithm to optimize the coordinates with. Read description to determine
        which one is best suited for yout application.

    Returns:
    --------
    mol: Chem.Mol
        Optimized molecule.

    prop: Any
        Property associated with the molecule.
    """
    logger.info("Optimizing coords for compound {0:d} using {1:s}: {2:s}...".format(idx, algo, Chem.MolToSmiles(m)))

    # Add H's to molecule (with 3D coords)
    mol = Chem.AddHs(m, addCoords=True)

    # Optimize and embed (write) coordinates using one of the defined force fields.
    if algo == "ETKDG":
        # Fast, and accurate conformation generator
        k = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if k != 0:
            return None, None

    elif algo == "UFF":
        # Universal Force Field
        AllChem.EmbedMultipleConfs(mol, 50, pruneRmsThresh=0.5, numThreads=0)
        try:
            arr = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000, numThreads=0)
        except ValueError:
            return None, None

        if not arr:
            return None, None
        else:
            arr = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000, numThreads=0)
            idx = np.argmin(arr, axis=0)[1] # get idx of lowest energy conformation
            conf = mol.GetConformers()[idx]
            mol.RemoveAllConformers()
            mol.AddConformer(conf)

    elif algo == "MMFF":
        # Merck Molecular Force Field
        AllChem.EmbedMultipleConfs(mol, 50, pruneRmsThresh=0.5, numThreads=0)
        try:
            arr = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000, numThreads=0)
        except ValueError:
            return None, None

        if not arr:
            return None, None
        else:
            arr = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000, numThreads=0)
            idx = np.argmin(arr, axis=0)[1] # get idx of lowest energy conformation
            conf = mol.GetConformers()[idx]
            mol.RemoveAllConformers()
            mol.AddConformer(conf)

    # Remove unwanted H's from molecule
    mol = Chem.RemoveHs(mol)
    return mol, prop


def convert(filepath: str,
            save_path: str, 
            x_name: str, 
            y_name: str, 
            n_workers: Optional[int]=mp.cpu_count()-1) -> None:
    """
    Load the SMILES representations of the molecules and process into final processed dataset.
    Contains optimized 3D coordinates of molecules.

    Params:
    -------
    filepath: str
        Path where the dataset is located.

    save_path: str
        Path to store modified data.

    x_name: str
        Column name of data X.

    y_name: str
        Column name of labels associated with the data X.

    n_workers: int, optional, default=1
        Number of processors to use in parallel to compute coordinates.
    """
    logger.info('Loading dataset from `{0:s}`...'.format(filepath))
    ext = os.path.splitext(filepath)[1]

    # Load dataset
    if ext == '.csv':
        X,y = load_csv(filepath, x_name=x_name, y_name=y_name)
        mols, props = [], []
        for smi, prop in zip(X, y):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mols.append(mol)
                props.append(prop)
        mol_idx = list(range(len(mols)))
    elif ext == '.sdf':
        mols = Chem.SDMolSupplier(filepath)
        props = [mol.GetProp(y_name) if mol.HasProp(y_name) else None for mol in mols]
        mol_idx = list(range(len(mols)))
    else:
        raise ValueError('Unsupported file type! Please provide a .csv or .sdf file.')
    logger.info('Successfully loaded {0:d} molecules from `{1:s}`'.format(len(mols), filepath))

    # Optimize 3D coordinates using multiprocessing. NOTE: This takes a long time.
    logger.info("Optimizing coordinates...")
    pool = mp.Pool(processes=n_workers)
    results = pool.starmap(func=optimize_coords, iterable=zip(mol_idx, mols, props))

    # Add properties associated with mols
    mol_list = []
    for mol, prop in results:
        if mol is not None:
            mol.SetProp(y_name, str(prop))
            mol_list.append(mol)
    logger.info('Optimized {0:d} molecules.'.format(len(mol_list)))

    # Create directory if unavailable
    save_path = os.path.expanduser(save_path)
    save_dir, _ = os.path.split(save_path)
    if not os.path.exists(save_dir):
        logger.info('Creating directory `{0:s}`'.format(save_dir))
        os.makedirs(save_dir)
    
    # Save coordinates to file
    writer = Chem.SDWriter(save_path)
    for m in mol_list: writer.write(m)
    writer.close()
    logger.info('Saved {0:d} molecules to `{1:s}`'.format(len(mol_list), save_path))
    