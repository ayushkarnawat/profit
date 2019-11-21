import pytest
import numpy as np

from rdkit.Chem import rdmolops, rdmolfiles, rdDistGeom
from profit.dataset.preprocessing import mol_feats


@pytest.fixture
def sample_mol():
    mol = rdmolfiles.MolFromSmiles('CN=C=O')
    mol = rdmolops.AddHs(mol, addCoords=True)
    rdDistGeom.EmbedMolecule(mol, rdDistGeom.ETKDG())
    return rdmolops.RemoveHs(mol)


@pytest.fixture
def sample_mol2():
    mol = rdmolfiles.MolFromSmiles('Cc1ccccc1')
    mol = rdmolops.AddHs(mol, addCoords=True)
    rdDistGeom.EmbedMolecule(mol, rdDistGeom.ETKDG())
    return rdmolops.RemoveHs(mol)


class TestGetAdjMatrix(object):

    def test_normal(self, sample_mol2):
        adj = mol_feats.construct_adj_matrix(sample_mol2, normalize=False)

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
        adj = mol_feats.construct_adj_matrix(sample_mol2, add_self_loops=False, normalize=False)

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
        adj = mol_feats.construct_adj_matrix(sample_mol2, out_size=8, add_self_loops=True, normalize=False)

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
            _ = mol_feats.construct_adj_matrix(sample_mol2, out_size=4)