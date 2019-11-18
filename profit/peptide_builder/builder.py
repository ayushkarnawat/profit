"""
Generate accurate 3D structure of peptide(s) using defined geometry 
(i.e. bond lengths, angles, and dihedral angles) information. Default 
geometry values are defined in the `geometry.py` file.
"""

import math
import numpy as np

from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.vectors import calc_dihedral, rotaxis, Vector

from typing import List, Union
from profit.peptide_builder.geometry import *


def _calculate_atom_coords(atom_a: Atom, atom_b: Atom, atom_c: Atom, length: float, 
                               angle: float, dihedral: float) -> np.ndarray:
    """Calculate cartesian coordinates of neighboring atom."""
    # Extract reference atom coordinates
    atom_a_coords = atom_a.get_vector()
    atom_b_coords = atom_b.get_vector()
    atom_c_coords = atom_c.get_vector()

    # Calculate C-apha and C-beta coordinates, and unpack
    Ca_X, Ca_Y, Ca_Z = tuple(atom_a_coords - atom_c_coords)
    Cb_X, Cb_Y, Cb_Z = tuple(atom_b_coords - atom_c_coords)
    
    # Plane parameters
    A = (Ca_Y*Cb_Z) - (Ca_Z*Cb_Y)
    B = (Ca_Z*Cb_X) - (Ca_X*Cb_Z)
    G = (Ca_X*Cb_Y) - (Ca_Y*Cb_X)

    # Constants
    # TODO: Needs better documentation/understanding
    F = math.sqrt(Cb_X*Cb_X + Cb_Y*Cb_Y + Cb_Z*Cb_Z) * length * math.cos(angle*(math.pi/180.0))
    const = math.sqrt(math.pow((B*Cb_Z - Cb_Y*G), 2) * (-(F*F)*(A*A+B*B+G*G)+(B*B*(Cb_X*Cb_X+Cb_Z*Cb_Z) 
                      + A*A*(Cb_Y*Cb_Y+Cb_Z*Cb_Z)- (2*A*Cb_X*Cb_Z*G) + (Cb_X*Cb_X+ Cb_Y*Cb_Y)*G*G 
                      - (2*B*Cb_Y)*(A*Cb_X+Cb_Z*G))*length*length))
    denom = (B*B)*(Cb_X*Cb_X+Cb_Z*Cb_Z)+ (A*A)*(Cb_Y*Cb_Y+Cb_Z*Cb_Z) - (2*A*Cb_X*Cb_Z*G) \
            + (Cb_X*Cb_X+Cb_Y*Cb_Y)*(G*G) - (2*B*Cb_Y)*(A*Cb_X+Cb_Z*G)

    # Calculate cartesian coords for new atom position
    # NOTE: This is based off the 3 reference atoms, bond length, angle, and diangle
    X = ((B*B*Cb_X*F)-(A*B*Cb_Y*F)+(F*G)*(-A*Cb_Z+Cb_X*G)+const)/denom
    if (B==0 or Cb_Z==0) and (Cb_Y==0 or G==0):
        const1 = math.sqrt(G*G*(-A*A*X*X+(B*B+G*G) * (length-X) * (length+X)))
        Y = ((-A*B*X)+const1) / (B*B+G*G)
        Z = -(A*G*G*X+B*const1) / (G*(B*B+G*G))
    else:
        Y = ((A*A*Cb_Y*F) * (B*Cb_Z-Cb_Y*G) + G*(-F*math.pow(B*Cb_Z-Cb_Y*G,2) + Cb_X*const) \
            - A*(B*B*Cb_X*Cb_Z*F - B*Cb_X*Cb_Y*F*G + Cb_Z*const)) / ((B*Cb_Z-Cb_Y*G)*denom)
        Z = ((A*A*Cb_Z*F) * (B*Cb_Z-Cb_Y*G) + (B*F)*math.pow(B*Cb_Z-Cb_Y*G,2) + (A*Cb_X*F*G) \
            * (-B*Cb_Z+Cb_Y*G) - B*Cb_X*const + A*Cb_Y*const) / ((B*Cb_Z-Cb_Y*G)*denom)

    # Create new vector (values computed with respect to the origin)
    atom_d_coords = Vector(X,Y,Z) + atom_c_coords
    new_dihedral = dihedral - calc_dihedral(atom_a_coords, atom_b_coords, 
                                            atom_c_coords, atom_d_coords) * (180.0/math.pi)
    rot = rotaxis(math.pi*(new_dihedral/180.0), atom_c_coords-atom_b_coords)
    atom_d_coords = (atom_d_coords - atom_b_coords).left_multiply(rot) + atom_b_coords
    return atom_d_coords.get_array()


def _create_residue(seg_id: int, geo: AminoAcidGeometry, atoms: List[Atom]):
    """Create residue with specified atoms."""
    res = Residue((" ", seg_id, " "), resname=geo.full_name, segid="    ")
    for atom in atoms: res.add(atom)
    return res
    

def make_gly(seg_id: int, geo: GlyGeometry, N: Atom, Ca: Atom, C: Atom, O: Atom) -> Residue:
    """Make glycine residue."""
    return _create_residue(seg_id, geo, atoms=[N, Ca, C, O])


def make_ala(seg_id: int, geo: AlaGeometry, N: Atom, Ca: Atom, C: Atom, O: Atom) -> Residue:
    # Calculate coords for R-group atom(s)
    Cb_coords = _calculate_atom_coords(N, Ca, C, geo.Ca_Cb_length, geo.C_Ca_Cb_angle, geo.N_C_Ca_Cb_diangle)
    Cb = Atom("CB", Cb_coords, 0.0, 1.0, " "," CB", 0, "C")
    return _create_residue(seg_id, geo, atoms=[N, Ca, C, O, Cb])


def make_ser(seg_id: int, geo: SerGeometry, N: Atom, Ca: Atom, C: Atom, O: Atom) -> Residue:
    # Calculate coords for R-group atom(s)
    Cb_coords = _calculate_atom_coords(N, Ca, C, geo.Ca_Cb_length, geo.C_Ca_Cb_angle, geo.N_C_Ca_Cb_diangle)
    Cb = Atom("CB", Cb_coords, 0.0, 1.0, " "," CB", 0, "C")
    Og_coords = _calculate_atom_coords(N, Ca, Cb, geo.Cb_Og_length, geo.Ca_Cb_Og_angle, geo.N_Ca_Cb_Og_diangle)
    Og = Atom("OG", Og_coords, 0.0, 1.0, " ", " OG", 0, "O")
    return _create_residue(seg_id, geo, atoms=[N, Ca, C, O, Cb, Og])


def make_cys(seg_id: int, geo: CysGeometry, N: Atom, Ca: Atom, C: Atom, O: Atom) -> Residue:
    # Calculate coords for R-group atom(s)
    Cb_coords = _calculate_atom_coords(N, Ca, C, geo.Ca_Cb_length, geo.C_Ca_Cb_angle, geo.N_C_Ca_Cb_diangle)
    Cb = Atom("CB", Cb_coords, 0.0, 1.0, " "," CB", 0, "C")
    Sg_coords = _calculate_atom_coords(N, Ca, Cb, geo.Cb_Sg_length, geo.Ca_Cb_Sg_angle, geo.N_Ca_Cb_Sg_diangle)
    Sg = Atom("SG", Sg_coords, 0.0, 1.0, " "," SG", 0, "S")
    return _create_residue(seg_id, geo, atoms=[N, Ca, C, O, Cb, Sg])


def make_val(seg_id: int, geo: ValGeometry, N: Atom, Ca: Atom, C: Atom, O: Atom) -> Residue:
    # Calculate coords for R-group atom(s)
    Cb_coords = _calculate_atom_coords(N, Ca, C, geo.Ca_Cb_length, geo.C_Ca_Cb_angle, geo.N_C_Ca_Cb_diangle)
    Cb = Atom("CB", Cb_coords, 0.0, 1.0, " "," CB", 0, "C")
    Cg1_coords = _calculate_atom_coords(N, Ca, Cb, geo.Cb_Cg1_length, geo.Ca_Cb_Cg1_angle, geo.N_Ca_Cb_Cg1_diangle)
    Cg1 = Atom("CG1", Cg1_coords, 0.0, 1.0, " "," CG1", 0, "C")
    Cg2_coords = _calculate_atom_coords(N, Ca, Cb, geo.Cb_Cg2_length, geo.Ca_Cb_Cg2_angle, geo.N_Ca_Cb_Cg2_diangle)
    Cg2 = Atom("CG2", Cg2_coords, 0.0, 1.0, " ", " CG2", 0, "C")
    return _create_residue(seg_id, geo, atoms=[N, Ca, C, O, Cb, Cg1, Cg2])
    
    
def make_ile(seg_id: int, geo: IleGeometry, N: Atom, Ca: Atom, C: Atom, O: Atom) -> Residue:
    # Calculate coords for R-group atom(s)
    Cb_coords = _calculate_atom_coords(N, Ca, C, geo.Ca_Cb_length, geo.C_Ca_Cb_angle, geo.N_C_Ca_Cb_diangle)
    Cb = Atom("CB", Cb_coords, 0.0, 1.0, " "," CB", 0, "C")
    Cg1_coords = _calculate_atom_coords(N, Ca, Cb, geo.Cb_Cg1_length, geo.Ca_Cb_Cg1_angle, geo.N_Ca_Cb_Cg1_diangle)
    Cg1 = Atom("CG1", Cg1_coords, 0.0, 1.0, " "," CG1", 0, "C")
    Cg2_coords = _calculate_atom_coords(N, Ca, Cb, geo.Cb_Cg2_length, geo.Ca_Cb_Cg2_angle, geo.N_Ca_Cb_Cg2_diangle)
    Cg2 = Atom("CG2", Cg2_coords, 0.0, 1.0, " ", " CG2", 0, "C")
    Cd1_coords = _calculate_atom_coords(Ca, Cb, Cg1, geo.Cg1_Cd1_length, geo.Cb_Cg1_Cd1_angle, geo.Ca_Cb_Cg1_Cd1_diangle) 
    Cd1 = Atom("CD1", Cd1_coords, 0.0, 1.0, " ", " CD1", 0, "C")
    return _create_residue(seg_id, geo, atoms=[N, Ca, C, O, Cb, Cg1, Cg2, Cd1])


def make_leu(seg_id: int, geo: LeuGeometry, N: Atom, Ca: Atom, C: Atom, O: Atom) -> Residue:
    # Calculate coords for R-group atom(s)
    Cb_coords = _calculate_atom_coords(N, Ca, C, geo.Ca_Cb_length, geo.C_Ca_Cb_angle, geo.N_C_Ca_Cb_diangle)
    Cb = Atom("CB", Cb_coords, 0.0, 1.0, " "," CB", 0, "C")
    Cg_coords = _calculate_atom_coords(N, Ca, Cb, geo.Cb_Cg_length, geo.Ca_Cb_Cg_angle, geo.N_Ca_Cb_Cg_diangle)
    Cg = Atom("CG", Cg_coords, 0.0, 1.0, " "," CG", 0, "C")
    Cd1_coords = _calculate_atom_coords(Ca, Cb, Cg, geo.Cg_Cd1_length, geo.Cb_Cg_Cd1_angle, geo.Ca_Cb_Cg_Cd1_diangle)
    Cd1 = Atom("CD1", Cd1_coords, 0.0, 1.0, " ", " CD1", 0, "C")
    Cd2_coords = _calculate_atom_coords(Ca, Cb, Cg, geo.Cg_Cd1_length, geo.Cb_Cg_Cd1_angle, geo.Ca_Cb_Cg_Cd1_diangle)
    Cd2 = Atom("CD2", Cd2_coords, 0.0, 1.0, " ", " CD2", 0, "C")
    return _create_residue(seg_id, geo, atoms=[N, Ca, C, O, Cb, Cg, Cd1, Cd2])


def make_thr(seg_id: int, geo: ThrGeometry, N: Atom, Ca: Atom, C: Atom, O: Atom) -> Residue:
    # Calculate coords for R-group atom(s)
    Cb_coords = _calculate_atom_coords(N, Ca, C, geo.Ca_Cb_length, geo.C_Ca_Cb_angle, geo.N_C_Ca_Cb_diangle)
    Cb = Atom("CB", Cb_coords, 0.0, 1.0, " "," CB", 0, "C")
    Og1_coords = _calculate_atom_coords(N, Ca, Cb, geo.Cb_Og1_length, geo.Ca_Cb_Og1_angle, geo.N_Ca_Cb_Og1_diangle)
    Og1 = Atom("OG", Og1_coords, 0.0, 1.0, " ", " OG1", 0, "O")
    Cg2_coords = _calculate_atom_coords(N, Ca, Cb, geo.Cb_Cg2_length, geo.Ca_Cb_Cg2_angle, geo.N_Ca_Cb_Cg2_diangle)
    Cg2 = Atom("CG2", Cg2_coords, 0.0, 1.0, " ", " CG2", 0, "C")
    return _create_residue(seg_id, geo, atoms=[N, Ca, C, O, Cb, Og1, Cg2])


def make_arg(seg_id: int, geo: ArgGeometry, N: Atom, Ca: Atom, C: Atom, O: Atom) -> Residue:
    # Calculate coords for R-group atom(s)
    Cb_coords = _calculate_atom_coords(N, Ca, C, geo.Ca_Cb_length, geo.C_Ca_Cb_angle, geo.N_C_Ca_Cb_diangle)
    Cb = Atom("CB", Cb_coords, 0.0, 1.0, " "," CB", 0, "C")
    Cg_coords = _calculate_atom_coords(N, Ca, Cb, geo.Cb_Cg_length, geo.Ca_Cb_Cg_angle, geo.N_Ca_Cb_Cg_diangle)
    Cg = Atom("CG", Cg_coords, 0.0, 1.0, " ", " CG", 0, "C")
    Cd_coords= _calculate_atom_coords(Ca, Cb, Cg, geo.Cg_Cd_length, geo.Cb_Cg_Cd_angle, geo.Ca_Cb_Cg_Cd_diangle)
    Cd = Atom("CD", Cd_coords, 0.0, 1.0, " ", " CD", 0, "C")
    Ne_coords = _calculate_atom_coords(Cb, Cg, Cd, geo.Cd_Ne_length, geo.Cg_Cd_Ne_angle, geo.Cb_Cg_Cd_Ne_diangle)
    Ne = Atom("NE", Ne_coords, 0.0, 1.0, " ", " NE", 0, "N")
    Cz_coords = _calculate_atom_coords(Cg, Cd, Ne, geo.Ne_Cz_length, geo.Cd_Ne_Cz_angle, geo.Cg_Cd_Ne_Cz_diangle)
    Cz = Atom("CZ", Cz_coords, 0.0, 1.0, " ", " CZ", 0, "C")
    Nh1_coords = _calculate_atom_coords(Cd, Ne, Cz, geo.Cz_Nh1_length, geo.Ne_Cz_Nh1_angle, geo.Cd_Ne_Cz_Nh1_diangle)
    Nh1 = Atom("NH1", Nh1_coords, 0.0, 1.0, " ", " NH1", 0, "N")
    Nh2_coords = _calculate_atom_coords(Cd, Ne, Cz, geo.Cz_Nh2_length, geo.Ne_Cz_Nh2_angle, geo.Cd_Ne_Cz_Nh2_diangle)
    Nh2 = Atom("NH2", Nh2_coords, 0.0, 1.0, " ", " NH2", 0, "N")
    return _create_residue(seg_id, geo, atoms=[N, Ca, C, O, Cb, Cg, Cd, Ne, Cz, Nh1, Nh2])


def make_lys(seg_id: int, geo: LysGeometry, N: Atom, Ca: Atom, C: Atom, O: Atom) -> Residue:
    # Calculate coords for R-group atom(s)
    Cb_coords = _calculate_atom_coords(N, Ca, C, geo.Ca_Cb_length, geo.C_Ca_Cb_angle, geo.N_C_Ca_Cb_diangle)
    Cb = Atom("CB", Cb_coords, 0.0, 1.0, " "," CB", 0, "C")
    Cg_coords = _calculate_atom_coords(N, Ca, Cb, geo.Cb_Cg_length, geo.Ca_Cb_Cg_angle, geo.N_Ca_Cb_Cg_diangle)
    Cg = Atom("CG", Cg_coords, 0.0, 1.0, " ", " CG", 0, "C")
    Cd_coords = _calculate_atom_coords(Ca, Cb, Cg, geo.Cg_Cd_length, geo.Cb_Cg_Cd_angle, geo.Ca_Cb_Cg_Cd_diangle)
    Cd = Atom("CD", Cd_coords, 0.0, 1.0, " ", " CD", 0, "C")
    Ce_coords = _calculate_atom_coords(Cb, Cg, Cd, geo.Cd_Ce_length, geo.Cg_Cd_Ce_angle, geo.Cb_Cg_Cd_Ce_diangle)
    Ce = Atom("CE", Ce_coords, 0.0, 1.0, " ", " CE", 0, "C")
    Nz_coords = _calculate_atom_coords(Cg, Cd, Ce, geo.Ce_Nz_length, geo.Cd_Ce_Nz_angle, geo.Cg_Cd_Ce_Nz_diangle)
    Nz = Atom("NZ", Nz_coords, 0.0, 1.0, " ", " NZ", 0, "N")
    return _create_residue(seg_id, geo, atoms=[N, Ca, C, O, Cb, Cg, Cd, Ce, Nz])


def make_asp(seg_id: int, geo: AspGeometry, N: Atom, Ca: Atom, C: Atom, O: Atom) -> Residue:
    # Calculate coords for R-group atom(s)
    Cb_coords = _calculate_atom_coords(N, Ca, C, geo.Ca_Cb_length, geo.C_Ca_Cb_angle, geo.N_C_Ca_Cb_diangle)
    Cb = Atom("CB", Cb_coords, 0.0, 1.0, " "," CB", 0, "C")
    Cg_coords = _calculate_atom_coords(N, Ca, Cb, geo.Cb_Cg_length, geo.Ca_Cb_Cg_angle, geo.N_Ca_Cb_Cg_diangle)
    Cg = Atom("CG", Cg_coords, 0.0, 1.0, " ", " CG", 0, "C")
    Od1_coords = _calculate_atom_coords(Ca, Cb, Cg, geo.Cg_Od1_length, geo.Cb_Cg_Od1_angle, geo.Ca_Cb_Cg_Od1_diangle)
    Od1 = Atom("OD1", Od1_coords, 0.0, 1.0, " ", " OD1", 0, "O")
    Od2_coords = _calculate_atom_coords(Ca, Cb, Cg, geo.Cg_Od2_length, geo.Cb_Cg_Od2_angle, geo.Ca_Cb_Cg_Od2_diangle)
    Od2 = Atom("OD2", Od2_coords, 0.0, 1.0, " ", " OD2", 0, "O")
    return _create_residue(seg_id, geo, atoms=[N, Ca, C, O, Cb, Cg, Od1, Od2])


def make_asn(seg_id: int, geo: AsnGeometry, N: Atom, Ca: Atom, C: Atom, O: Atom) -> Residue:
    # Calculate coords for R-group atom(s)
    Cb_coords = _calculate_atom_coords(N, Ca, C, geo.Ca_Cb_length, geo.C_Ca_Cb_angle, geo.N_C_Ca_Cb_diangle)
    Cb = Atom("CB", Cb_coords, 0.0, 1.0, " "," CB", 0, "C")
    Cg_coords = _calculate_atom_coords(N, Ca, Cb, geo.Cb_Cg_length, geo.Ca_Cb_Cg_angle, geo.N_Ca_Cb_Cg_diangle)
    Cg = Atom("CG", Cg_coords, 0.0, 1.0, " ", " CG", 0, "C")
    Od1_coords = _calculate_atom_coords(Ca, Cb, Cg, geo.Cg_Od1_length, geo.Cb_Cg_Od1_angle, geo.Ca_Cb_Cg_Od1_diangle)
    Od1 = Atom("OD1", Od1_coords, 0.0, 1.0, " ", " OD1", 0, "O")
    Nd2_coords = _calculate_atom_coords(Ca, Cb, Cg, geo.Cg_Nd2_length, geo.Cb_Cg_Nd2_angle, geo.Ca_Cb_Cg_Nd2_diangle)
    Nd2 = Atom("ND2", Nd2_coords, 0.0, 1.0, " ", " ND2", 0, "N")
    return _create_residue(seg_id, geo, atoms=[N, Ca, C, O, Cb, Cg, Od1, Nd2])


def make_glu(seg_id: int, geo: GluGeometry, N: Atom, Ca: Atom, C: Atom, O: Atom) -> Residue:
    # Calculate coords for R-group atom(s)
    Cb_coords = _calculate_atom_coords(N, Ca, C, geo.Ca_Cb_length, geo.C_Ca_Cb_angle, geo.N_C_Ca_Cb_diangle)
    Cb = Atom("CB", Cb_coords, 0.0, 1.0, " "," CB", 0, "C")
    Cg_coords = _calculate_atom_coords(N, Ca, Cb, geo.Cb_Cg_length, geo.Ca_Cb_Cg_angle, geo.N_Ca_Cb_Cg_diangle)
    Cg = Atom("CG", Cg_coords, 0.0, 1.0, " ", " CG", 0, "C")
    Cd_coords = _calculate_atom_coords(Ca, Cb, Cg, geo.Cg_Cd_length, geo.Cb_Cg_Cd_angle, geo.Ca_Cb_Cg_Cd_diangle)
    Cd = Atom("CD", Cd_coords, 0.0, 1.0, " ", " CD", 0, "C")
    Oe1_coords = _calculate_atom_coords(Cb, Cg, Cd, geo.Cd_Oe1_length, geo.Cg_Cd_Oe1_angle, geo.Cb_Cg_Cd_Oe1_diangle)
    Oe1= Atom("OE1", Oe1_coords, 0.0, 1.0, " ", " OE1", 0, "O")
    Oe2_coords = _calculate_atom_coords(Cb, Cg, Cd, geo.Cd_Oe2_length, geo.Cg_Cd_Oe2_angle, geo.Cb_Cg_Cd_Oe2_diangle)
    Oe2 = Atom("OE2", Oe2_coords, 0.0, 1.0, " ", " OE2", 0, "O")
    return _create_residue(seg_id, geo, atoms=[N, Ca, C, O, Cb, Cg, Cd, Oe1, Oe2])


def make_gln(seg_id: int, geo: GlnGeometry, N: Atom, Ca: Atom, C: Atom, O: Atom) -> Residue:
    # Calculate coords for R-group atom(s)
    Cb_coords = _calculate_atom_coords(N, Ca, C, geo.Ca_Cb_length, geo.C_Ca_Cb_angle, geo.N_C_Ca_Cb_diangle)
    Cb = Atom("CB", Cb_coords, 0.0, 1.0, " "," CB", 0, "C")
    Cg_coords = _calculate_atom_coords(N, Ca, Cb, geo.Cb_Cg_length, geo.Ca_Cb_Cg_angle, geo.N_Ca_Cb_Cg_diangle)
    Cg = Atom("CG", Cg_coords, 0.0, 1.0, " ", " CG", 0, "C")
    Cd_coords = _calculate_atom_coords(Ca, Cb, Cg, geo.Cg_Cd_length, geo.Cb_Cg_Cd_angle, geo.Ca_Cb_Cg_Cd_diangle)
    Cd = Atom("CD", Cd_coords, 0.0, 1.0, " ", " CD", 0, "C")
    Oe1_coords = _calculate_atom_coords(Cb, Cg, Cd, geo.Cd_Oe1_length, geo.Cg_Cd_Oe1_angle, geo.Cb_Cg_Cd_Oe1_diangle)
    Oe1 = Atom("OE1", Oe1_coords, 0.0, 1.0, " ", " OE1", 0, "O")
    Ne2_coords = _calculate_atom_coords(Cb, Cg, Cd, geo.Cd_Ne2_length, geo.Cg_Cd_Ne2_angle, geo.Cb_Cg_Cd_Ne2_diangle)
    Ne2 = Atom("NE2", Ne2_coords, 0.0, 1.0, " ", " NE2", 0, "N")
    return _create_residue(seg_id, geo, atoms=[N, Ca, C, O, Cb, Cg, Cd, Oe1, Ne2])


def make_met(seg_id: int, geo: MetGeometry, N: Atom, Ca: Atom, C: Atom, O: Atom) -> Residue:
    # Calculate coords for R-group atom(s)
    Cb_coords = _calculate_atom_coords(N, Ca, C, geo.Ca_Cb_length, geo.C_Ca_Cb_angle, geo.N_C_Ca_Cb_diangle)
    Cb = Atom("CB", Cb_coords, 0.0, 1.0, " "," CB", 0, "C")
    Cg_coords = _calculate_atom_coords(N, Ca, Cb, geo.Cb_Cg_length, geo.Ca_Cb_Cg_angle, geo.N_Ca_Cb_Cg_diangle)
    Cg = Atom("CG", Cg_coords, 0.0, 1.0, " ", " CG", 0, "C")
    Sd_coords = _calculate_atom_coords(Ca, Cb, Cg, geo.Cg_Sd_length, geo.Cb_Cg_Sd_angle, geo.Ca_Cb_Cg_Sd_diangle)
    Sd = Atom("SD", Sd_coords, 0.0, 1.0, " ", " SD", 0, "S")
    Ce_coords = _calculate_atom_coords(Cb, Cg, Sd, geo.Sd_Ce_length, geo.Cg_Sd_Ce_angle, geo.Cb_Cg_Sd_Ce_diangle)
    Ce = Atom("CE", Ce_coords, 0.0, 1.0, " ", " CE", 0, "C")
    return _create_residue(seg_id, geo, atoms=[N, Ca, C, O, Cb, Cg, Sd, Ce])


def make_his(seg_id: int, geo: HisGeometry, N: Atom, Ca: Atom, C: Atom, O: Atom) -> Residue:
    # Calculate coords for R-group atom(s)
    Cb_coords = _calculate_atom_coords(N, Ca, C, geo.Ca_Cb_length, geo.C_Ca_Cb_angle, geo.N_C_Ca_Cb_diangle)
    Cb = Atom("CB", Cb_coords, 0.0, 1.0, " "," CB", 0, "C")
    Cg_coords = _calculate_atom_coords(N, Ca, Cb, geo.Cb_Cg_length, geo.Ca_Cb_Cg_angle, geo.N_Ca_Cb_Cg_diangle)
    Cg = Atom("CG", Cg_coords, 0.0, 1.0, " ", " CG", 0, "C")
    Nd1_coords = _calculate_atom_coords(Ca, Cb, Cg, geo.Cg_Nd1_length, geo.Cb_Cg_Nd1_angle, geo.Ca_Cb_Cg_Nd1_diangle)
    Nd1 = Atom("ND1", Nd1_coords, 0.0, 1.0, " ", " ND1", 0, "N")
    Cd2_coords = _calculate_atom_coords(Ca, Cb, Cg, geo.Cg_Cd2_length, geo.Cb_Cg_Cd2_angle, geo.Ca_Cb_Cg_Cd2_diangle)
    Cd2 = Atom("CD2", Cd2_coords, 0.0, 1.0, " ", " CD2", 0, "C")
    Ce1_coords = _calculate_atom_coords(Cb, Cg, Nd1, geo.Nd1_Ce1_length, geo.Cg_Nd1_Ce1_angle, geo.Cb_Cg_Nd1_Ce1_diangle)
    Ce1 = Atom("CE1", Ce1_coords, 0.0, 1.0, " ", " CE1", 0, "C")
    Ne2_coords = _calculate_atom_coords(Cb, Cg, Cd2, geo.Cd2_Ne2_length, geo.Cg_Cd2_Ne2_angle, geo.Cb_Cg_Cd2_Ne2_diangle)
    Ne2 = Atom("NE2", Ne2_coords, 0.0, 1.0, " ", " NE2", 0, "N")
    return _create_residue(seg_id, geo, atoms=[N, Ca, C, O, Cb, Cg, Nd1, Cd2, Ce1, Ne2])


def make_pro(seg_id: int, geo: ProGeometry, N: Atom, Ca: Atom, C: Atom, O: Atom) -> Residue:
    # Calculate coords for R-group atom(s)
    Cb_coords = _calculate_atom_coords(N, Ca, C, geo.Ca_Cb_length, geo.C_Ca_Cb_angle, geo.N_C_Ca_Cb_diangle)
    Cb = Atom("CB", Cb_coords, 0.0, 1.0, " "," CB", 0, "C")
    Cg_coords = _calculate_atom_coords(N, Ca, Cb, geo.Cb_Cg_length, geo.Ca_Cb_Cg_angle, geo.N_Ca_Cb_Cg_diangle)
    Cg = Atom("CG", Cg_coords, 0.0, 1.0, " ", " CG", 0, "C")
    Cd_coords = _calculate_atom_coords(Ca, Cb, Cg, geo.Cg_Cd_length, geo.Cb_Cg_Cd_angle, geo.Ca_Cb_Cg_Cd_diangle)
    Cd = Atom("CD", Cd_coords, 0.0, 1.0, " ", " CD", 0, "C")
    return _create_residue(seg_id, geo, atoms=[N, Ca, C, O, Cb, Cg, Cd])


def make_phe(seg_id: int, geo: PheGeometry, N: Atom, Ca: Atom, C: Atom, O: Atom) -> Residue:
    # Calculate coords for R-group atom(s)
    Cb_coords = _calculate_atom_coords(N, Ca, C, geo.Ca_Cb_length, geo.C_Ca_Cb_angle, geo.N_C_Ca_Cb_diangle)
    Cb = Atom("CB", Cb_coords, 0.0, 1.0, " "," CB", 0, "C")
    Cg_coords = _calculate_atom_coords(N, Ca, Cb, geo.Cb_Cg_length, geo.Ca_Cb_Cg_angle, geo.N_Ca_Cb_Cg_diangle)
    Cg = Atom("CG", Cg_coords, 0.0, 1.0, " ", " CG", 0, "C")
    Cd1_coords = _calculate_atom_coords(Ca, Cb, Cg, geo.Cg_Cd1_length, geo.Cb_Cg_Cd1_angle, geo.Ca_Cb_Cg_Cd1_diangle)
    Cd1 = Atom("CD1", Cd1_coords, 0.0, 1.0, " ", " CD1", 0, "C")
    Cd2_coords = _calculate_atom_coords(Ca, Cb, Cg, geo.Cg_Cd2_length, geo.Cb_Cg_Cd2_angle, geo.Ca_Cb_Cg_Cd2_diangle)
    Cd2 = Atom("CD2", Cd2_coords, 0.0, 1.0, " ", " CD2", 0, "C")
    Ce1_coords = _calculate_atom_coords(Cb, Cg, Cd1, geo.Cd1_Ce1_length, geo.Cg_Cd1_Ce1_angle, geo.Cb_Cg_Cd1_Ce1_diangle)
    Ce1 = Atom("CE1", Ce1_coords, 0.0, 1.0, " ", " CE1", 0, "C")
    Ce2_coords = _calculate_atom_coords(Cb, Cg, Cd2, geo.Cd2_Ce2_length, geo.Cg_Cd2_Ce2_angle, geo.Cb_Cg_Cd2_Ce2_diangle)
    Ce2 = Atom("CE2", Ce2_coords, 0.0, 1.0, " ", " CE2", 0, "C")
    Cz_coords = _calculate_atom_coords(Cg, Cd1, Ce1, geo.Ce1_Cz_length, geo.Cd1_Ce1_Cz_angle, geo.Cg_Cd1_Ce1_Cz_diangle)
    Cz = Atom("CZ", Cz_coords, 0.0, 1.0, " ", " CZ", 0, "C")
    return _create_residue(seg_id, geo, atoms=[N, Ca, C, O, Cb, Cg, Cd1, Cd2, Ce1, Ce2, Cz])


def make_tyr(seg_id: int, geo: TyrGeometry, N: Atom, Ca: Atom, C: Atom, O: Atom) -> Residue:
    # Calculate coords for R-group atom(s)
    Cb_coords = _calculate_atom_coords(N, Ca, C, geo.Ca_Cb_length, geo.C_Ca_Cb_angle, geo.N_C_Ca_Cb_diangle)
    Cb = Atom("CB", Cb_coords, 0.0, 1.0, " "," CB", 0, "C")
    Cg_coords = _calculate_atom_coords(N, Ca, Cb, geo.Cb_Cg_length, geo.Ca_Cb_Cg_angle, geo.N_Ca_Cb_Cg_diangle)
    Cg = Atom("CG", Cg_coords, 0.0, 1.0, " ", " CG", 0, "C")
    Cd1_coords = _calculate_atom_coords(Ca, Cb, Cg, geo.Cg_Cd1_length, geo.Cb_Cg_Cd1_angle, geo.Ca_Cb_Cg_Cd1_diangle)
    Cd1 = Atom("CD1", Cd1_coords, 0.0, 1.0, " ", " CD1", 0, "C")
    Cd2_coords = _calculate_atom_coords(Ca, Cb, Cg, geo.Cg_Cd2_length, geo.Cb_Cg_Cd2_angle, geo.Ca_Cb_Cg_Cd2_diangle)
    Cd2 = Atom("CD2", Cd2_coords, 0.0, 1.0, " ", " CD2", 0, "C")
    Ce1_coords = _calculate_atom_coords(Cb, Cg, Cd1, geo.Cd1_Ce1_length, geo.Cg_Cd1_Ce1_angle, geo.Cb_Cg_Cd1_Ce1_diangle)
    Ce1 = Atom("CE1", Ce1_coords, 0.0, 1.0, " ", " CE1", 0, "C")
    Ce2_coords = _calculate_atom_coords(Cb, Cg, Cd2, geo.Cd2_Ce2_length, geo.Cg_Cd2_Ce2_angle, geo.Cb_Cg_Cd2_Ce2_diangle)
    Ce2 = Atom("CE2", Ce2_coords, 0.0, 1.0, " ", " CE2", 0, "C")
    Cz_coords = _calculate_atom_coords(Cg, Cd1, Ce1, geo.Ce1_Cz_length, geo.Cd1_Ce1_Cz_angle, geo.Cg_Cd1_Ce1_Cz_diangle)
    Cz = Atom("CZ", Cz_coords, 0.0, 1.0, " ", " CZ", 0, "C")
    Oh_coords = _calculate_atom_coords(Cd1, Ce1, Cz, geo.Cz_OH_length, geo.Ce1_Cz_OH_angle, geo.Cd1_Ce1_Cz_OH_diangle)
    Oh = Atom("OH", Oh_coords, 0.0, 1.0, " ", " OH", 0, "O")
    return _create_residue(seg_id, geo, atoms=[N, Ca, C, O, Cb, Cg, Cd1, Cd2, Ce1, Ce2, Cz, Oh])


def make_trp(seg_id: int, geo: TrpGeometry, N: Atom, Ca: Atom, C: Atom, O: Atom) -> Residue:
    # Calculate coords for R-group atom(s)
    Cb_coords = _calculate_atom_coords(N, Ca, C, geo.Ca_Cb_length, geo.C_Ca_Cb_angle, geo.N_C_Ca_Cb_diangle)
    Cb = Atom("CB", Cb_coords, 0.0, 1.0, " "," CB", 0, "C")
    Cg_coords = _calculate_atom_coords(N, Ca, Cb, geo.Cb_Cg_length, geo.Ca_Cb_Cg_angle, geo.N_Ca_Cb_Cg_diangle)
    Cg = Atom("CG", Cg_coords, 0.0, 1.0, " ", " CG", 0, "C")
    Cd1_coords = _calculate_atom_coords(Ca, Cb, Cg, geo.Cg_Cd1_length, geo.Cb_Cg_Cd1_angle, geo.Ca_Cb_Cg_Cd1_diangle)
    Cd1 = Atom("CD1", Cd1_coords, 0.0, 1.0, " ", " CD1", 0, "C")
    Cd2_coords = _calculate_atom_coords(Ca, Cb, Cg, geo.Cg_Cd2_length, geo.Cb_Cg_Cd2_angle, geo.Ca_Cb_Cg_Cd2_diangle)
    Cd2 = Atom("CD2", Cd2_coords, 0.0, 1.0, " ", " CD2", 0, "C")
    Ne1_coords = _calculate_atom_coords(Cb, Cg, Cd1, geo.Cd1_Ne1_length, geo.Cg_Cd1_Ne1_angle, geo.Cb_Cg_Cd1_Ne1_diangle)
    Ne1 = Atom("NE1", Ne1_coords, 0.0, 1.0, " ", " NE1", 0, "N")
    Ce2_coords = _calculate_atom_coords(Cb, Cg, Cd2, geo.Cd2_Ce2_length, geo.Cg_Cd2_Ce2_angle, geo.Cb_Cg_Cd2_Ce2_diangle)
    Ce2 = Atom("CE2", Ce2_coords, 0.0, 1.0, " ", " CE2", 0, "C")
    Ce3_coords = _calculate_atom_coords(Cb, Cg, Cd2, geo.Cd2_Ce3_length, geo.Cg_Cd2_Ce3_angle, geo.Cb_Cg_Cd2_Ce3_diangle)
    Ce3 = Atom("CE3", Ce3_coords, 0.0, 1.0, " ", " CE3", 0, "C")
    Cz2_coords = _calculate_atom_coords(Cg, Cd2, Ce2, geo.Cd2_Ce2_length, geo.Cd2_Ce2_Cz2_angle, geo.Cg_Cd2_Ce2_Cz2_diangle)
    Cz2 = Atom("CZ2", Cz2_coords, 0.0, 1.0, " ", " CZ2", 0, "C")
    Cz3_coords = _calculate_atom_coords(Cg, Cd2, Ce3, geo.Ce3_Cz3_length, geo.Cd2_Ce3_Cz3_angle, geo.Cg_Cd2_Ce3_Cz3_diangle)
    Cz3 = Atom("CZ3", Cz3_coords, 0.0, 1.0, " ", " CZ3", 0, "C")
    Ch2_coords = _calculate_atom_coords(Cd2, Ce2, Cz2, geo.Cz2_CH2_length, geo.Ce2_Cz2_CH2_angle, geo.Cd2_Ce2_Cz2_CH2_diangle)
    Ch2 = Atom("CH2", Ch2_coords, 0.0, 1.0, " ", " CH2", 0, "C")
    return _create_residue(seg_id, geo, atoms=[N, Ca, C, O, Cb, Cg, Cd1, Cd2, Ne1, Ce2, Ce3, Cz2, Cz3, Ch2]) 


def make_full_residue(seg_id, geo, N, Ca, C, O):
    resname = geo.residue_name
    if resname == "G":
        residue = make_gly(seg_id, geo, N, Ca, C, O)
    elif resname == "A":
        residue = make_ala(seg_id, geo, N, Ca, C, O)
    elif resname == "S":
        residue = make_ser(seg_id, geo, N, Ca, C, O)
    elif resname == "C":
        residue = make_cys(seg_id, geo, N, Ca, C, O)
    elif resname == "V":
        residue = make_val(seg_id, geo, N, Ca, C, O)
    elif resname == "I":
        residue = make_ile(seg_id, geo, N, Ca, C, O)
    elif resname == "L":
        residue = make_leu(seg_id, geo, N, Ca, C, O)
    elif resname == "T":
        residue = make_thr(seg_id, geo, N, Ca, C, O)
    elif resname == "R":
        residue = make_arg(seg_id, geo, N, Ca, C, O)
    elif resname == "K":
        residue = make_lys(seg_id, geo, N, Ca, C, O)
    elif resname == "D":
        residue = make_asp(seg_id, geo, N, Ca, C, O)
    elif resname == "E":
        residue = make_glu(seg_id, geo, N, Ca, C, O)
    elif resname == "N":
        residue = make_asn(seg_id, geo, N, Ca, C, O)
    elif resname == "Q":
        residue = make_gln(seg_id, geo, N, Ca, C, O)
    elif resname == "M":
        residue = make_met(seg_id, geo, N, Ca, C, O)
    elif resname == "H":
        residue = make_his(seg_id, geo, N, Ca, C, O)
    elif resname == "P":
        residue = make_pro(seg_id, geo, N, Ca, C, O)
    elif resname == "F":
        residue = make_phe(seg_id, geo, N, Ca, C, O)
    elif resname == "Y":
        residue = make_tyr(seg_id, geo, N, Ca, C, O)
    elif resname == "W":
        residue = make_trp(seg_id, geo, N, Ca, C, O)
    return residue


class PDBResidueBuilder(object):
    """Build PDB structure from defined residue geometries."""


    def __init__(self, id: Union[str, int]):
        self.structure_builder = StructureBuilder()
        self.structure_builder.init_structure(id)


    def get_structure(self):
        return self.structure_builder.get_structure()


    def _get_model(self, model_id: int) -> Model:
        return self.get_structure()[model_id]


    def add(self, residue: Union[Residue, AminoAcidGeometry, str], model_id: int=0, chain_id: Union[str, int]="A"):
        """Add residue to structure/model/chain.

        If first residue, then just use coordinates (if it already has them), otherwise start @ origin.
        If in the middle, use the previous coordinates as the reference.
        """
        # Create model + chain if they don't exist
        try:
            model = self._get_model(model_id)
        except KeyError:
            self.structure_builder.init_model(model_id)
            model = self._get_model(model_id)

        try:
            chain: Chain = model[chain_id]
        except KeyError:
            self.structure_builder.init_chain(chain_id)
            chain: Chain = model[chain_id]

        geo = None
        if not isinstance(residue, Residue):
            geo = residue if isinstance(residue, AminoAcidGeometry) else aa_geometries[residue] 
        
        # If there are current no residues in the chain, add passed in residue
        # either with defined coordinates or origin (if it doesn't have them).
        if len(list(chain.get_residues())) == 0:
            if geo is not None:
                seg_id = 1

                # Compute backbone coordinates
                Ca_coords = np.array([0,0,0], dtype=np.float)
                C_coords = np.array([geo.Ca_C_length,0,0], dtype=np.float)
                N_coords = np.array([geo.Ca_N_length*math.cos(geo.N_Ca_C_angle*(math.pi/180.0)), 
                                     geo.Ca_N_length*math.sin(geo.N_Ca_C_angle*(math.pi/180.0)), 
                                     0], dtype=np.float)
                Ca =Atom("CA", Ca_coords, 0.0, 1.0, " ", " CA", 0, "C")
                C = Atom("C", C_coords, 0.0, 1.0, " ", " C", 0, "C")
                N = Atom("N", N_coords, 0.0, 1.0, " ", " N", 0, "N")
                
                O_coords = _calculate_atom_coords(N, Ca, C, geo.C_O_length, geo.Ca_C_O_angle, geo.N_Ca_C_O_diangle)
                O = Atom("O", O_coords , 0.0, 1.0, " ", " O", 0, "O")
                residue = make_full_residue(seg_id, geo, N, Ca, C, O)
            chain.add(residue)
        else:
            if geo is not None:
                ref_residue: Residue = list(chain.get_residues())[-1]
                seg_id = ref_residue.get_id()[1] + 1

                # Compute backbone coordinates
                N_coords = _calculate_atom_coords(ref_residue['N'], ref_residue['CA'], 
                                                  ref_residue['C'], geo.peptide_bond, 
                                                  geo.Ca_C_N_angle, geo.psi)
                N = Atom("N", N_coords, 0.0, 1.0, " ", " N", 0, "N")

                Ca_coords = _calculate_atom_coords(ref_residue['CA'], ref_residue['C'], 
                                                   N, geo.Ca_N_length, geo.C_N_Ca_angle, 
                                                   geo.omega)
                Ca = Atom("CA", Ca_coords, 0.0, 1.0, " ", " CA", 0, "C")

                C_coords = _calculate_atom_coords(ref_residue["C"], N, Ca, geo.Ca_C_length, 
                                                  geo.N_Ca_C_angle, geo.phi)
                C = Atom("C", C_coords, 0.0, 1.0, " ", " C", 0, "C")

                # Create C=O (Carbonyl) group, to be moved later
                O_coords = _calculate_atom_coords(N, Ca, C, geo.C_O_length, geo.Ca_C_O_angle, 
                                                  geo.N_Ca_C_O_diangle)
                O = Atom("O", O_coords, 0.0 , 1.0, " ", " O", 0, "O")
                
                # Update coordinates to reflect new backbone structure
                residue = make_full_residue(seg_id, geo, N, Ca, C, O)
                ref_residue["O"].set_coord(_calculate_atom_coords(residue["N"], ref_residue["CA"], 
                                                                  ref_residue["C"], geo.C_O_length, 
                                                                  geo.Ca_C_O_angle, 180.0))
                residue["O"].set_coord(_calculate_atom_coords(residue['N'], residue['CA'], residue['C'], 
                                                              geo.C_O_length, geo.Ca_C_O_angle, 180.0))
            
            chain.add(residue)


if __name__ == "__main__":
    from Bio.PDB.PDBParser import PDBParser
    from Bio.PDB.PDBIO import PDBIO
    from Bio.PDB.Polypeptide import three_to_one
    structure = PDBParser().get_structure('3gb1', 'data/pdb/pdb3gb1.ent')
    residues: List[Residue] = list(structure[0]['A'].get_residues())
    # ala = AlaGeometry()
    # residue: Residue = structure[0]['A'][1]
    # atoms: List[Atom] = [atom for atom in residue.get_atoms()]
    # D = _calculate_atom_coords(atoms[0], atoms[1], atoms[2], ala.Ca_Cb_length, 
    #                            ala.C_Ca_Cb_angle, ala.N_C_Ca_Cb_diangle)

    # Modify residue at defined positions
    replace_with = {39: 'T', 40: 'N', 41: 'T', 54: 'Y'}
    rb = PDBResidueBuilder('test')
    for residue in residues:
        current_resseq = residue.get_id()[1]
        if current_resseq in replace_with.keys():
            rb.add(replace_with.get(current_resseq))
        else:
            rb.add(three_to_one(residue.resname.upper()))
            # rb.add(residue)
    structure = rb.get_structure()
    
    # Save
    io = PDBIO()
    io.set_structure(structure)
    io.save('data/raw/mutated2.pdb')