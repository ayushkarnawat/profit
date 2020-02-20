import os

import pytest
import numpy as np

from rdkit.Chem import rdDistGeom, rdmolfiles, rdmolops

from profit.dataset.parsers.sdf_parser import SDFFileParser
from profit.dataset.preprocessors.egcn_preprocessor import EGCNPreprocessor
from profit.utils.io_utils import maybe_create_dir


@pytest.fixture
def test_mols():
    mols = []
    all_smiles = ['CN=C=O', 'Cc1ccccc1', 'CC1=CC2CC(CC1)O2', 'CCCCCCCCCCCCCCCC']
    for smiles in all_smiles:
        mol = rdmolfiles.MolFromSmiles(smiles)
        mol = rdmolops.AddHs(mol, addCoords=True)
        rdDistGeom.EmbedMolecule(mol, rdDistGeom.ETKDG())
        mol = rdmolops.RemoveHs(mol)
        mol.SetProp('Fitness', str(np.random.rand(1)[0]))
        mols.append(mol)
    return mols


@pytest.fixture()
def sdf_file(test_mols):
    # Create directory for test file(s)
    tmp_dir = maybe_create_dir('data/tmp/')
    fname = os.path.join(tmp_dir, 'test.sdf')
    # Store molecules
    writer = rdmolfiles.SDWriter(fname)
    for mol in test_mols: 
        writer.write(mol)
    writer.close()
    return fname


def test_sdf_file_parser_not_return_smiles(sdf_file, test_mols):
    preprocessor = EGCNPreprocessor(max_atoms=49, out_size=49)
    parser = SDFFileParser(preprocessor)
    result = parser.parse(sdf_file, return_smiles=False)
    dataset = result['dataset']
    smiles = result['smiles']
    assert len(dataset) == 3
    assert smiles is None

    # Check if computed features are saved correctly
    for i in range(len(dataset)): # for each feature
        for j in range(len(test_mols)): # and for each example
            expect = preprocessor.get_input_feats(test_mols[j])
            np.testing.assert_array_almost_equal(dataset[i][j], expect[i], decimal=3)


def test_sdf_file_parser_return_smiles(sdf_file, test_mols):
    preprocessor = EGCNPreprocessor(max_atoms=49, out_size=49)
    parser = SDFFileParser(preprocessor)
    result = parser.parse(sdf_file, return_smiles=True)
    dataset = result['dataset']
    smiles = result['smiles']
    assert len(dataset) == 3

    # Check if computed features are saved correctly
    for i in range(len(dataset)): # for each feature
        for j in range(len(test_mols)): # and for each example
            expect = preprocessor.get_input_feats(test_mols[j])
            np.testing.assert_array_almost_equal(dataset[i][j], expect[i], decimal=3)
    
    # Check smiles array
    assert type(smiles) == np.ndarray
    assert smiles.ndim == 1
    assert len(smiles) == dataset[0].shape[0]
    expected_smiles = np.array([rdmolfiles.MolToSmiles(mol) for mol in test_mols])
    np.testing.assert_array_equal(smiles, expected_smiles)


def test_sdf_file_parser_target_index(sdf_file, test_mols):
    idxs = [0,2]
    preprocessor = EGCNPreprocessor(max_atoms=49, out_size=49)
    parser = SDFFileParser(preprocessor, labels='Fitness')
    result = parser.parse(sdf_file, return_smiles=True, target_index=idxs)
    dataset = result['dataset']
    smiles = result['smiles']
    assert len(dataset) == 4
    
    # # Check if computed features are saved correctly
    for i in range(len(dataset)-1): # for each feature
        for data_idx, j in enumerate(idxs): # and for each example
            expect = preprocessor.get_input_feats(test_mols[j])
            np.testing.assert_array_almost_equal(dataset[i][data_idx], expect[i], decimal=3)

    # Check if labels are parsed correctly
    labels = dataset[3]
    expected_labels = np.array([preprocessor.get_labels(test_mols[idx], 'Fitness') for idx in idxs])
    np.testing.assert_array_almost_equal(labels, expected_labels, decimal=3)

    # Check smiles array
    assert type(smiles) == np.ndarray
    assert smiles.ndim == 1
    assert len(smiles) == dataset[0].shape[0]
    expected_smiles = np.array([rdmolfiles.MolToSmiles(test_mols[idx]) for idx in idxs])
    np.testing.assert_array_equal(smiles, expected_smiles)
