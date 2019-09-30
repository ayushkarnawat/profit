
from typing import Any, List, Optional

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops


HYDROGEN_DONOR = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
HYROGEN_ACCEPTOR = Chem.MolFromSmarts("[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S" + 
                                      ";-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([" + 
                                      "o,s]:n);!$([o,s]:c:n)])]")
ACIDIC = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
BASIC = Chem.MolFromSmarts("[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!" + 
                           "$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])" + 
                           "([C;!$(C(=O))])[C;!$(C(=O))])]")


class MolFeatureExtractionError(Exception):
    pass


def one_hot(x: Any, allowable_set: List[Any]) -> List[int]:
    """One hot encode labels. 

    If label `x` is not included in the set, set the value to the last element in the list.
    TODO: Should the above statement be hold? How else can we use the last elem in the list.

    Params:
    -------
    x: Any
        Label to one hot encode.

    allowed_set: list of Any
        All possible values the label can have. 

    Returns:
    --------
    vec: list of int
        One hot encoded vector of the features with the label `x` as the True label.

    Examples:
    ---------
    >>> one_hot(x='Si', allowable_set=['C', 'O', 'N', 'S', 'Cl', 'F', 'Br', 'P', 
                                       'I', 'Si', 'B', 'Na', 'Sn', 'Se', 'other'])
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    """
    # Use last index of set if x is not in set
    if x not in allowable_set:
        x = allowable_set[:-1]
    return list(map(lambda s: int(x == s), allowable_set))


def check_num_atoms(mol: Chem.Mol, max_num_atoms: Optional[int]=-1):
    """Check number of atoms in `mol` does not exceed `max_num_atoms`.

    If number of atoms in `mol` exceeds the number `max_num_atoms`, it will
    raise `MolFeatureExtractionError` exception.

    Params:
    -------
    mol: rdkit.Chem.Mol
        The molecule to check.
        
    num_max_atoms: int, optional , default=-1 
        Maximum allowed number of atoms in a molecule. If negative, check passes unconditionally.
    """
    num_atoms = mol.GetNumAtoms()
    if max_num_atoms >= 0 and num_atoms > max_num_atoms:
        raise MolFeatureExtractionError('Atoms in mol (N={}) exceeds num_max_atoms (N={})' \
            .format(num_atoms, max_num_atoms))


def construct_mol_features(mol: Chem.Mol, out_size: Optional[int]=-1) -> np.ndarray:
    """Returns the atom features of all the atoms in the molecule.
    
    Params:
    -------
    mol: rdkit.Chem.Mol
        Molecule of interest. 

    out_size: int, optional, default=-1
        The size of the returned array. If this option is negative, it does not take any effect.
        Otherwise, it must be larger than or equal to the number of atoms in the input molecule. 
        If so, the end of the array is padded with zeros.

    Returns:
    --------
    mol_feats: np.ndarray, shape=(n, m)
        Where `n` is the total number of atoms within the molecule, and `m` is the number of feats.
    """
    # Caluclate charges and chirality of atoms within molecule
    AllChem.ComputeGasteigerCharges(mol) # stored under _GasteigerCharge
    Chem.AssignStereochemistry(mol) # stored under _CIPCode, see doc for more info

    # Retrieve atom index locations of matches
    hydrogen_donor_match = sum(mol.GetSubstructMatches(HYDROGEN_DONOR), ())
    hydrogen_acceptor_match = sum(mol.GetSubstructMatches(HYROGEN_ACCEPTOR), ())
    acidic_match = sum(mol.GetSubstructMatches(ACIDIC), ())
    basic_match = sum(mol.GetSubstructMatches(BASIC), ())

    # Get ring info
    ring = mol.GetRingInfo()

    mol_feats = []
    n_atoms = mol.GetNumAtoms()
    for atom_idx in range(n_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)

        atom_feats = []
        atom_feats += one_hot(atom.GetSymbol(), ['C', 'O', 'N', 'S', 'Cl', 'F', 'Br', 'P', 
                                                 'I', 'Si', 'B', 'Na', 'Sn', 'Se', 'other'])
        atom_feats += one_hot(atom.GetDegree(), [1,2,3,4,5,6])
        atom_feats += one_hot(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                        Chem.rdchem.HybridizationType.SP2,
                                                        Chem.rdchem.HybridizationType.SP3,
                                                        Chem.rdchem.HybridizationType.SP3D,
                                                        Chem.rdchem.HybridizationType.SP3D2])
        atom_feats += one_hot(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])
        atom_feats += one_hot(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3])
        atom_feats += [atom.GetProp("_GasteigerCharge")]
        atom_feats += [atom.GetIsAromatic()]

        atom_feats += [ring.IsAtomInRingOfSize(atom_idx, 3),
                       ring.IsAtomInRingOfSize(atom_idx, 4),
                       ring.IsAtomInRingOfSize(atom_idx, 5),
                       ring.IsAtomInRingOfSize(atom_idx, 6),
                       ring.IsAtomInRingOfSize(atom_idx, 7),
                       ring.IsAtomInRingOfSize(atom_idx, 8)]
        atom_feats += one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

        # Chirality
        try:
            atom_feats += one_hot(atom.GetProp('_CIPCode'), ["R", "S"]) + [atom.HasProp("_ChiralityPossible")]
        except:
            atom_feats += [False, False] + [atom.HasProp("_ChiralityPossible")]
        # Hydrogen bonding
        atom_feats += [atom_idx in hydrogen_donor_match]
        atom_feats += [atom_idx in hydrogen_acceptor_match]
        # Is Acidic/Basic
        atom_feats += [atom_idx in acidic_match]
        atom_feats += [atom_idx in basic_match]

        mol_feats.append(atom_feats)

    if out_size < 0:
        return np.array(mol_feats, dtype=np.float)
    elif out_size >= n_atoms:
        # 'empty' padding for `mol_feats`. Useful to generate feature matrix of same size for all mols
        # NOTE: len(mol_feats[0]) is the number of feats
        padded_mol_feats = np.zeros((out_size, len(mol_feats[0])), dtype=np.float)
        padded_mol_feats[:n_atoms] = np.array(mol_feats, dtype=np.float)
        return padded_mol_feats
    else:
        raise ValueError('`out_size` (N={}) must be negative or larger than or equal to the '
                         'number of atoms in the input molecules (N={}).'.format(out_size, n_atoms))
    

def construct_adj_matrix(mol: Chem.Mol, 
                         out_size: Optional[int]=-1, 
                         add_self_loops: Optional[bool]=True,
                         normalize: Optional[bool]=True) -> np.ndarray:
    """Returns the adjacency matrix of the molecule.

    Normalization of the matrix is highly recommened. When we apply a layer propogation rule 
    defined by, 

    .. ::math: `f(H^{(l)}, A) = \\sigma(A H^{(l)} W^{(l)})

    multiplication with `A` will completely change the scale of the features vectors, which we can 
    observe by looking into the eigenvalues of A. By performing :math: `D^{-1}A`, where `D` is the 
    diagonal degree node matrix, the rows become normalized to 1. However, in practice, it is 
    better to use symmetric normalization (i.e. :math:`D^{-\\frac{1/2}} \\hat{A} D^{-\\frac{1/2}})
    as that has been observed to yield better results.  

    Additionally, when multiplying by `A`, for every node, we sum up all the feature vectors of all 
    neighboring nodes but not the node itself (unless there are self-loops in the graph). We can 
    "fix" this by adding self-loops in the graph: aka add an identity matrix `I` to `A`.

    See https://tkipf.github.io/graph-convolutional-networks/ for more in-depth overview. 

    Params:
    -------
    mol: rdkit.Chem.Mol
        Molecule of interest.

    out_size: int, optional, default=-1
        The size of the returned array. If this option is negative, it does not take any effect.
        Otherwise, it must be larger than or equal to the number of atoms in the input molecule. 
        If so, the end of the array is padded with zeros.

    add_self_loops: bool, optional, default=True
        Whether or not to add the `I` matrix (aka self-connections). If normalize is True, this 
        option is ignored.

    normalize: bool, optional, default=True
        Whether or not to normalize the matrix. If `True`, the diagonal elements are filled with 1,
        and symmetric normalization is performed: :math:`D^{-\\frac{1/2}} * \\hat{A} * D^{-\\frac{1/2}}`

    Returns:
    --------
    adj: np.ndarray
        Adjacency matrix of input molecule. If ``out_size`` is non-negative, the returned matrix 
        is equal to that value. Otherwise, it is equal to the number of atoms in the the molecule.
    """
    adj = rdmolops.GetAdjacencyMatrix(mol)
    s1, s2 = adj.shape # shape=(n_atoms, n_atoms) 

    # Normalize using D^(-1/2) * A_hat * D^(-1/2)
    if normalize:
        adj = adj + np.eye(s1)
        degree = np.array(adj.sum(1))
        deg_inv_sqrt = np.power(degree, -0.5)
        deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.
        deg_inv_sqrt = np.diag(deg_inv_sqrt)
        adj = deg_inv_sqrt
    elif add_self_loops:
        adj = adj + np.eye(s1)
    
    if out_size < 0:
        return adj
    elif out_size >= s1:
        # 'empty' padding for `adj`. Useful to generate adj matrix of same size for all mols
        padded_adj = np.zeros(shape=(out_size, out_size), dtype=np.float)
        padded_adj[:s1, :s2] = adj
        return padded_adj
    else:
        raise ValueError('`out_size` (N={}) must be negative or larger than or equal to the '
                         'number of atoms in the input molecules (N={}).'.format(out_size, s1))


def construct_pos_matrix(mol: Chem.Mol, out_size: Optional[int]=-1) -> np.ndarray:
    """Construct relative positions from each atom within the molecule.

    Params:
    -------
    mol: rdkit.Chem.Mol
        Molecule of interest. 

    out_size: int, optional, default=-1
        The size of the returned array. If this option is negative, it does not take any effect.
        Otherwise, it must be larger than or equal to the number of atoms in the input molecule. 
        If so, the end of the array is padded with zeros.

    Returns:
    --------
    pos_matrix: np.ndarray, shape=(n,n,3)
        Relative position (XYZ) coordinates from one atom the others in the mol. 

    Examples:
    ---------
    >>> smiles = 'N[C@@]([H])([C@]([H])(O2)C)C(=O)N[C@@]([H])(CC(=O)N)C(=O)N[C@@]([H])([C@]([H])' \
                 '(O)C)C(=O)N[C@@]([H])(Cc1ccc(O)cc1)C(=O)2'
    >>> mol = Chem.MolFromSmiles(smiles)
    >>> mol = Chem.AddHs(mol, addCoords=True)
    >>> AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    >>> mol = Chem.RemoveHs(mol)
    >>> pos_matrix = construct_pos_matrix(mol, out_size=-1)
    >>> pos_matrix.shape
    (34,34,3)

    >>> pos_matrix = construct_pos_matrix(mol, out_size=49)
    >>> pos_matrix.shape
    (49,49,3)
    """
    N = mol.GetNumAtoms()
    coords = mol.GetConformer().GetPositions() # shape=(N,3)

    # Determine appropiate output size to generate feature matrix of same size for all mols.
    if out_size < 0:
        size = N
    elif out_size > N:
        size = out_size
    else:
        raise ValueError('`out_size` (N={}) is smaller than number of atoms in mol (N={})'.
                         format(out_size, N))
    
    pos_matrix = np.zeros(shape=(size, size, 3), dtype=np.float)
    for atom_idx in range(N):
        atom_pos = coords[atom_idx] # central atom of interest
        for neighbor_idx in range(N):
            neigh_pos = coords[neighbor_idx] # neighboring atom
            pos_matrix[atom_idx, neighbor_idx] = atom_pos - neigh_pos # dist between neighbor -> center
    return pos_matrix
