import pytest
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from profit.dataset.preprocessors import common


@pytest.fixture
def sample_mol():
    mol = Chem.MolFromSmiles('CN=C=O')
    mol = Chem.AddHs(mol, addCoords=True)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    return Chem.RemoveHs(mol)


@pytest.fixture
def sample_mol2():
    mol = Chem.MolFromSmiles('Cc1ccccc1')
    mol = Chem.AddHs(mol, addCoords=True)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    return Chem.RemoveHs(mol)


class TestGetAdjMatrix(object):

    def test_normal(self, sample_mol2):
        adj = common.construct_adj_matrix(sample_mol2, normalize=False)

        assert adj.shape == (7, 7)
        expect = np.array([[1., 1., 0., 0., 0., 0., 0.],
                           [1., 1., 1., 0., 0., 0., 1.],
                           [0., 1., 1., 1., 0., 0., 0.],
                           [0., 0., 1., 1., 1., 0., 0.],
                           [0., 0., 0., 1., 1., 1., 0.],
                           [0., 0., 0., 0., 1., 1., 1.],
                           [0., 1., 0., 0., 0., 1., 1.]], dtype=np.float32)
        np.testing.assert_equal(adj, expect)

    def test_normal_no_self_connection(self, sample_mol2):
        adj = common.construct_adj_matrix(sample_mol2, add_self_loops=False, normalize=False)

        assert adj.shape == (7, 7)
        expect = np.array([[0., 1., 0., 0., 0., 0., 0.],
                           [1., 0., 1., 0., 0., 0., 1.],
                           [0., 1., 0., 1., 0., 0., 0.],
                           [0., 0., 1., 0., 1., 0., 0.],
                           [0., 0., 0., 1., 0., 1., 0.],
                           [0., 0., 0., 0., 1., 0., 1.],
                           [0., 1., 0., 0., 0., 1., 0.]], dtype=np.float32)
        np.testing.assert_equal(adj, expect)

    def test_normal_padding(self, sample_mol2):
        adj = common.construct_adj_matrix(sample_mol2, out_size=8, add_self_loops=True, normalize=False)

        assert adj.shape == (8, 8)
        expect = np.array([[1., 1., 0., 0., 0., 0., 0., 0.],
                           [1., 1., 1., 0., 0., 0., 1., 0.],
                           [0., 1., 1., 1., 0., 0., 0., 0.],
                           [0., 0., 1., 1., 1., 0., 0., 0.],
                           [0., 0., 0., 1., 1., 1., 0., 0.],
                           [0., 0., 0., 0., 1., 1., 1., 0.],
                           [0., 1., 0., 0., 0., 1., 1., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0.]], dtype=np.float32)
        np.testing.assert_equal(adj, expect)

    def test_normal_truncated(self, sample_mol2):
        with pytest.raises(ValueError):
            adj = common.construct_adj_matrix(sample_mol2, out_size=4)