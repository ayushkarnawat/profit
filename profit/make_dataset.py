"""
Converts dataset into usable/processed form.
"""
import os
import csv
import logging
import multiprocessing as mp

from typing import Tuple, Optional, Any


import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

from profit.cyclops import struct_gen as sg

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s  %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def load_csv(filepath: str, 
             x_name: str, 
             y_name: str, 
             use_pd: Optional[bool]=True) -> Tuple[int, int]:
    """
    Loads the raw dataset.

    Params:
    -------
    filepath: str
        Path where the dataset is located.

    x_name: str
        Column name of the actual data X.

    y_name: str
        Column name of the labels in the dataset.

    Returns:
    --------
    X: np.ndarray
        Extracted data.

    y: np.ndarray
        Extracted labels corresponding to the dataset X. 
    """
    if use_pd:
        df = pd.read_csv(filepath, sep=',')
        X = np.array(df[x_name].values, dtype=str)
        y = np.array(df[y_name].values, dtype=float)
    else:
        X,y = [], []
        with open(filepath) as file:
            reader = csv.DictReader(file)
            for row in reader:
                X.append(row[x_name])
                y.append(row[y_name])
        X = np.array(X, dtype=str)
        y = np.array(y, dtype=float)
    return X,y


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
    X,y = load_csv(filepath, x_name=x_name, y_name=y_name, use_pd=True)

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
                    algo: Optional[str]="MMFF") -> Tuple[Chem.Mol, Any]:
    """
    Optimize 3D coordinates for each compound. Defines the XYZ positions for each atom in the 
    molecule. 

    In order to get accurate 3D conformations, it is a good idea to add H's to molecule first, 
    embed (write) and optimize the coordinates, and then remove the unwanted H's. 

    Params:
    -------
    idx: int
        Index location of molecule within dataset.
    
    m: Chem.Mol
        The molecule to optimize.
    
    prop: Any
        Property associated with the molecule.

    algo: str, optional, default='MMFF'
        Which force field algorithm to optimize the coordinates with.

    Returns:
    --------
    mol: Chem.Mol
        Optimized molecule.

    prop: Any
        Property associated with the molecule.
    """
    logger.info("Optimizing coords for compound {0:d}: {1:s}...".format(idx, Chem.MolToSmiles(m)))

    # Add H's to molecule
    mol = Chem.AddHs(m)

    # Optimize and embed (write) coordinates using one of the defined force fields.
    if algo == "ETKDG":
        # Landrum et al. DOI: 10.1021/acs.jcim.5b00654
        k = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if k != 0:
            return None, None

    elif algo == "UFF":
        # Universal Force Field
        AllChem.EmbedMultipleConfs(mol, 50, pruneRmsThresh=0.5)
        try:
            arr = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000)
        except ValueError:
            return None, None

        if not arr:
            return None, None
        else:
            arr = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=20000)
            idx = np.argmin(arr, axis=0)[1]
            conf = mol.GetConformers()[idx]
            mol.RemoveAllConformers()
            mol.AddConformer(conf)

    elif algo == "MMFF":
        # Merck Molecular Force Field
        AllChem.EmbedMultipleConfs(mol, 50, pruneRmsThresh=0.5)
        try:
            arr = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
        except ValueError:
            return None, None

        if not arr:
            return None, None
        else:
            arr = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=20000)
            idx = np.argmin(arr, axis=0)[1]
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
        X,y = load_csv(filepath, x_name=x_name, y_name=y_name, use_pd=True)
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
    logger.info('Saved {0:d} molecules to `{1:s}`'.format(len(mol_list), save_path))
    

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(prog=sys.argv[0], 
                                     description='Converts raw dataset to processed for modeling.',
                                     argument_default=argparse.SUPPRESS)
    parser.add_argument('--raw_path', metavar='RAW DATASET PATH', type=str,
                        help='Relative path to the dataset', required=True, default='data/raw/vdgv570.csv')
    parser.add_argument('--interim_path', metavar='INTERIM DATA PATH', type=str,
                        help='Relative path to save the interim dataset', default=None)
    parser.add_argument('--processed_path', metavar='PROCESSED DATA PATH', type=str,
                        help='Relative path to save the final, processed dataset', default=None)
    parser.add_argument('-x', '--x_name', metavar='DATA', type=str, help='Column name of dataset X', 
                        default='Variants')
    parser.add_argument('-y', '--y_name', metavar='LABEL', type=str,
                        help='Column name of labels associated with the data X', default='Fitness')
    parser.add_argument('-c', '--constraints', action='store_true',
                        help='Save molecules with constraints or not', default=False)
    parser.add_argument('-n', '--n_workers', metavar='NUM WORKERS', type=int,
                        help='Number of workers', default=mp.cpu_count()-1)
    args = vars(parser.parse_args())

    # If the interim and processed filepaths are not provided, use default args
    raw_fp = args['raw_path']
    filename, ext = os.path.splitext(raw_fp)
    split_file = filename.split('/')
    if args['interim_path'] is None:
        args['interim_path'] = split_file[0] + '/interim/' + split_file[-1] + '_smiles' + ext
    if args['processed_path'] is None:
        args['processed_path'] = split_file[0] + '/processed/' + split_file[-1] + '.sdf'
    
    # X,y = load_csv(args['raw_path'], x_name=args['x_name'], y_name=args['y_name'], use_pd=True)
    df = convert_to_smiles(args['raw_path'], args['interim_path'], args['x_name'], args['y_name'], 
                           save_constraints=args['constraints'])
    convert(args['interim_path'], args['processed_path'], x_name='SMILES', y_name=args['y_name'], 
            n_workers=args['n_workers'])
    