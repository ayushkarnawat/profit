"""Geometric properties of amino acids (AA)."""

import random

from abc import ABC
from typing import List, Union


class AminoAcidGeometry(ABC):
    """Base class for amino acid geometry."""

    def __init__(self):
        # Bond lengths
        self.Ca_N_length = 1.46
        self.Ca_C_length = 1.52
        self.C_O_length = 1.23

        # Atom angles
        self.Ca_C_N_angle = 116.642992978143
        self.C_N_Ca_angle = 121.382215820277

        # Misc properties
        self.omega = 180.0          # dihedral angle of right-hand rotation around C-N bond
        self.phi = -120.0           # dihedral angle of right-hand rotation around N-Ca bond
        self.psi = 140.0            # dihedral angle of right-hand rotation around Ca-C bond 
        self.peptide_bond = 1.33    # amide bond stength between amino acids

        # Compute number of "extra" dihedral angles
        # NOTE: There are atmost 2 common dihedral angles shared among amino acids. 
        # NOTE: Number is counted by seeing how many 'diangle' vars exits. THIS IS QUITE HACKY.
        common_diangles = ['N_Ca_C_O_diangle', 'N_C_Ca_Cb_diangle']
        all_diangles = [var for var in self.__dict__.keys() if 'diangle' in var.lower()]
        self.extra_diangles = list(filter(lambda angle: angle not in common_diangles, all_diangles))


    def set_rotamer_dihedrals(self, angles: Union[str, List[float]]="random") -> None:
        """Rotamer's dihedral angles.
        
        Params:
        -------
        angles: list of float or str, optional, default="random"
            Dihedral angles of rotamers. Should be of same length as the number of rotamers 
            other than N-CA--C-O, N-C--CA-CB, where '--' is the location of the dihedral angle.
            If "random", then random dihedral angles are chosen for the rotamers.
        """
        # Choose random angles for each "extra" rotamer
        if angles == "random":
            rotamer_angles = [-60, 60, 180]
            angles = [random.choice(rotamer_angles) for _ in range(0, len(self.extra_diangles))]

        if len(angles) != len(self.extra_diangles):
            raise ValueError("The number of provided rotamer angles must be {:d}. Got {:d} "
                             "instead.".format(len(self.extra_diangles), len(angles)))
        # Set new values 
        for idx, var in enumerate(self.extra_diangles):
            self.__dict__[var] = angles[idx]
    
    
    def __repr__(self):
        # Exclude member functions, only print variables and their values
        out = ["{0:} = {1:}".format(var, value) for var, value in self.__dict__.items()]
        return "\n".join(out)


class GlyGeometry(AminoAcidGeometry):
    """Glycine geometry."""

    def __init__(self):
        self.residue_name = "G"
        self.full_name = "Gly"

        # Amino acid specific angles and dihedrals
        self.N_Ca_C_angle = 110.8914
        self.Ca_C_O_angle = 120.5117
        self.N_Ca_C_O_diangle = 180.0

        # Add common/shared amino acid geometric properties. 
        # NOTE: This should always be last, as the computation of the # of dihedral angles relies 
        # on the number of 'dihedral' in the variable names. AGAIN, THIS IS HACKY.
        super(GlyGeometry, self).__init__()
    

class AlaGeometry(AminoAcidGeometry):
    """Alanine geometry."""

    def __init__(self):
        self.residue_name = "A"
        self.full_name = "Ala"

        # Amino acid specific bonds, angles, and dihedrals
        self.N_Ca_C_angle = 111.068
        self.Ca_C_O_angle = 120.5
        self.N_Ca_C_O_diangle = -60.5

        self.Ca_Cb_length = 1.52
        self.C_Ca_Cb_angle = 109.5
        self.N_C_Ca_Cb_diangle = 122.6860

        # Add common/shared amino acid geometric properties. 
        # NOTE: This should always be last, as the computation of the # of dihedral angles relies 
        # on the number of 'dihedral' in the variable names. AGAIN, THIS IS HACKY.
        super(AlaGeometry, self).__init__()


class SerGeometry(AminoAcidGeometry):
    """Serine geometry."""

    def __init__(self):
        self.residue_name = "S"
        self.full_name = "Ser"

        # Amino acid specific bonds, angles, and dihedrals
        self.N_Ca_C_angle = 111.2812
        self.Ca_C_O_angle = 120.5
        self.N_Ca_C_O_diangle = -60.0

        self.Ca_Cb_length = 1.52
        self.C_Ca_Cb_angle = 109.5
        self.N_C_Ca_Cb_diangle = 122.6618

        self.Cb_Og_length = 1.417
        self.Ca_Cb_Og_angle = 110.773
        self.N_Ca_Cb_Og_diangle = -63.3

        # Add common/shared amino acid geometric properties. 
        # NOTE: This should always be last, as the computation of the # of dihedral angles relies 
        # on the number of 'dihedral' in the variable names. AGAIN, THIS IS HACKY.
        super(SerGeometry, self).__init__()


class CysGeometry(AminoAcidGeometry):
    """Cystine geometry."""

    def __init__(self):
        self.residue_name = "C"
        self.full_name = "Cys"

        # Amino acid specific bonds, angles, and dihedrals
        self.N_Ca_C_angle = 110.8856 
        self.Ca_C_O_angle = 120.5
        self.N_Ca_C_O_diangle = -60.0

        self.Ca_Cb_length = 1.52
        self.C_Ca_Cb_angle = 109.5
        self.N_C_Ca_Cb_diangle = 122.5037

        self.Cb_Sg_length = 1.808
        self.Ca_Cb_Sg_angle = 113.8169
        self.N_Ca_Cb_Sg_diangle = -62.2

        # Add common/shared amino acid geometric properties. 
        # NOTE: This should always be last, as the computation of the # of dihedral angles relies 
        # on the number of 'dihedral' in the variable names. AGAIN, THIS IS HACKY.
        super(CysGeometry, self).__init__()


class ValGeometry(AminoAcidGeometry):
    """Valine geometry."""

    def __init__(self):
        self.residue_name = "V"
        self.full_name = "Val"

        # Amino acid specific bonds, angles, and dihedrals
        self.N_Ca_C_angle = 109.7698
        self.Ca_C_O_angle = 120.5686
        self.N_Ca_C_O_diangle = -60.0

        self.Ca_Cb_length = 1.52
        self.C_Ca_Cb_angle = 109.5
        self.N_C_Ca_Cb_diangle = 123.2347

        self.Cb_Cg1_length = 1.527
        self.Ca_Cb_Cg1_angle = 110.7
        self.N_Ca_Cb_Cg1_diangle = 177.2

        self.Cb_Cg2_length = 1.527
        self.Ca_Cb_Cg2_angle = 110.4
        self.N_Ca_Cb_Cg2_diangle = -63.3

        # Add common/shared amino acid geometric properties. 
        # NOTE: This should always be last, as the computation of the # of dihedral angles relies 
        # on the number of 'dihedral' in the variable names. AGAIN, THIS IS HACKY.
        super(ValGeometry, self).__init__()


class IleGeometry(AminoAcidGeometry):
    """Isoleucine geometry."""

    def __init__(self):
        self.residue_name = "I"
        self.full_name = "Ile"

        # Amino acid specific bonds, angles, and dihedrals
        self.N_Ca_C_angle = 109.7202
        self.Ca_C_O_angle = 120.5403
        self.N_Ca_C_O_diangle = -60.0

        self.Ca_Cb_length = 1.52
        self.C_Ca_Cb_angle = 109.5
        self.N_C_Ca_Cb_diangle = 123.2347

        self.Cb_Cg1_length = 1.527
        self.Ca_Cb_Cg1_angle = 110.7
        self.N_Ca_Cb_Cg1_diangle = 59.7

        self.Cb_Cg2_length = 1.527
        self.Ca_Cb_Cg2_angle = 110.4
        self.N_Ca_Cb_Cg2_diangle = -61.6

        self.Cg1_Cd1_length = 1.52
        self.Cb_Cg1_Cd1_angle = 113.97
        self.Ca_Cb_Cg1_Cd1_diangle = 169.8

        # Add common/shared amino acid geometric properties. 
        # NOTE: This should always be last, as the computation of the # of dihedral angles relies 
        # on the number of 'dihedral' in the variable names. AGAIN, THIS IS HACKY.
        super(IleGeometry, self).__init__()


class LeuGeometry(AminoAcidGeometry):
    """Leucine geometry."""

    def __init__(self):
        self.residue_name = "L"
        self.full_name = "Leu"

        # Amino acid specific bonds, angles, and dihedrals
        self.N_Ca_C_angle = 110.8652 
        self.Ca_C_O_angle = 120.4647
        self.N_Ca_C_O_diangle = 120.0

        self.Ca_Cb_length = 1.52
        self.C_Ca_Cb_angle = 109.5
        self.N_C_Ca_Cb_diangle = 122.4948

        self.Cb_Cg_length = 1.53
        self.Ca_Cb_Cg_angle = 116.10
        self.N_Ca_Cb_Cg_diangle = -60.1

        self.Cg_Cd1_length = 1.524
        self.Cb_Cg_Cd1_angle = 110.27
        self.Ca_Cb_Cg_Cd1_diangle = 174.9

        self.Cg_Cd2_length = 1.525
        self.Cb_Cg_Cd2_angle = 110.58
        self.Ca_Cb_Cg_Cd2_diangle = 66.7

        # Add common/shared amino acid geometric properties. 
        # NOTE: This should always be last, as the computation of the # of dihedral angles relies 
        # on the number of 'dihedral' in the variable names. AGAIN, THIS IS HACKY.
        super(LeuGeometry, self).__init__()


class ThrGeometry(AminoAcidGeometry):
    """Threonine geometry."""

    def __init__(self):
        self.residue_name = "T"
        self.full_name = "Thr"

        # Amino acid specific bonds, angles, and dihedrals
        self.N_Ca_C_angle = 110.7014
        self.Ca_C_O_angle = 120.5359
        self.N_Ca_C_O_diangle = 120.0

        self.Ca_Cb_length = 1.52
        self.C_Ca_Cb_angle = 109.5
        self.N_C_Ca_Cb_diangle = 123.0953

        self.Cb_Og1_length = 1.43
        self.Ca_Cb_Og1_angle = 109.18
        self.N_Ca_Cb_Og1_diangle = 60.0

        self.Cb_Cg2_length = 1.53
        self.Ca_Cb_Cg2_angle = 111.13
        self.N_Ca_Cb_Cg2_diangle = -60.3

        # Add common/shared amino acid geometric properties. 
        # NOTE: This should always be last, as the computation of the # of dihedral angles relies 
        # on the number of 'dihedral' in the variable names. AGAIN, THIS IS HACKY.
        super(ThrGeometry, self).__init__()


class ArgGeometry(AminoAcidGeometry):
    """Arginine geometry."""

    def __init__(self):
        self.residue_name = "R"
        self.full_name = "Arg"

        # Amino acid specific bonds, angles, and dihedrals
        self.N_Ca_C_angle = 110.98
        self.Ca_C_O_angle = 120.54
        self.N_Ca_C_O_diangle = 120.0

        self.Ca_Cb_length = 1.52
        self.C_Ca_Cb_angle = 109.5
        self.N_C_Ca_Cb_diangle = 122.76

        self.Cb_Cg_length = 1.52
        self.Ca_Cb_Cg_angle = 113.83
        self.N_Ca_Cb_Cg_diangle = -65.2

        self.Cg_Cd_length = 1.52
        self.Cb_Cg_Cd_angle = 111.79
        self.Ca_Cb_Cg_Cd_diangle = -179.2

        self.Cd_Ne_length = 1.46
        self.Cg_Cd_Ne_angle = 111.68
        self.Cb_Cg_Cd_Ne_diangle = -179.3

        self.Ne_Cz_length = 1.33
        self.Cd_Ne_Cz_angle = 124.79
        self.Cg_Cd_Ne_Cz_diangle = -178.7

        self.Cz_Nh1_length = 1.33
        self.Ne_Cz_Nh1_angle = 120.64
        self.Cd_Ne_Cz_Nh1_diangle = 0.0

        self.Cz_Nh2_length = 1.33
        self.Ne_Cz_Nh2_angle = 119.63
        self.Cd_Ne_Cz_Nh2_diangle = 180.0

        # Add common/shared amino acid geometric properties. 
        # NOTE: This should always be last, as the computation of the # of dihedral angles relies 
        # on the number of 'dihedral' in the variable names. AGAIN, THIS IS HACKY.
        super(ArgGeometry, self).__init__()


class LysGeometry(AminoAcidGeometry):
    """Lysine geometry."""

    def __init__(self):
        self.residue_name = "K"
        self.full_name = "Lys"

        # Amino acid specific bonds, angles, and dihedrals
        self.N_Ca_C_angle = 111.08
        self.Ca_C_O_angle = 120.54
        self.N_Ca_C_O_diangle = 120.0

        self.Ca_Cb_length = 1.52
        self.C_Ca_Cb_angle = 109.5
        self.N_C_Ca_Cb_diangle = 122.76

        self.Cb_Cg_length = 1.52
        self.Ca_Cb_Cg_angle = 113.83
        self.N_Ca_Cb_Cg_diangle = -64.5

        self.Cg_Cd_length = 1.52
        self.Cb_Cg_Cd_angle = 111.79
        self.Ca_Cb_Cg_Cd_diangle = -178.1

        self.Cd_Ce_length = 1.46
        self.Cg_Cd_Ce_angle = 111.68
        self.Cb_Cg_Cd_Ce_diangle = -179.6

        self.Ce_Nz_length = 1.33
        self.Cd_Ce_Nz_angle = 124.79
        self.Cg_Cd_Ce_Nz_diangle = 179.6

        # Add common/shared amino acid geometric properties. 
        # NOTE: This should always be last, as the computation of the # of dihedral angles relies 
        # on the number of 'dihedral' in the variable names. AGAIN, THIS IS HACKY.
        super(LysGeometry, self).__init__()


class AspGeometry(AminoAcidGeometry):
    """Aspartic acid geometry."""

    def __init__(self):
        self.residue_name = "D"
        self.full_name = "Asp"

        # Amino acid specific bonds, angles, and dihedrals
        self.N_Ca_C_angle = 111.03
        self.Ca_C_O_angle = 120.51
        self.N_Ca_C_O_diangle = 120.0

        self.Ca_Cb_length = 1.52
        self.C_Ca_Cb_angle = 109.5
        self.N_C_Ca_Cb_diangle = 122.82

        self.Cb_Cg_length = 1.52
        self.Ca_Cb_Cg_angle = 113.06
        self.N_Ca_Cb_Cg_diangle = -66.4

        self.Cg_Od1_length = 1.25
        self.Cb_Cg_Od1_angle = 119.22
        self.Ca_Cb_Cg_Od1_diangle = -46.7

        self.Cg_Od2_length = 1.25
        self.Cb_Cg_Od2_angle = 118.218
        self.Ca_Cb_Cg_Od2_diangle = 180 + self.Ca_Cb_Cg_Od1_diangle

        # Add common/shared amino acid geometric properties. 
        # NOTE: This should always be last, as the computation of the # of dihedral angles relies 
        # on the number of 'dihedral' in the variable names. AGAIN, THIS IS HACKY.
        super(AspGeometry, self).__init__()


class AsnGeometry(AminoAcidGeometry):
    """Asparagine geometry."""

    def __init__(self):
        self.residue_name = "N"
        self.full_name = "Asn"

        # Amino acid specific bonds, angles, and dihedrals
        self.N_Ca_C_angle = 111.5
        self.Ca_C_O_angle = 120.4826
        self.N_Ca_C_O_diangle = -60.0

        self.Ca_Cb_length = 1.52
        self.C_Ca_Cb_angle = 109.5
        self.N_C_Ca_Cb_diangle = 123.2254

        self.Cb_Cg_length = 1.52
        self.Ca_Cb_Cg_angle = 112.62
        self.N_Ca_Cb_Cg_diangle = -65.5

        self.Cg_Od1_length = 1.23
        self.Cb_Cg_Od1_angle = 120.85
        self.Ca_Cb_Cg_Od1_diangle = -58.3

        self.Cg_Nd2_length = 1.33
        self.Cb_Cg_Nd2_angle = 116.48
        self.Ca_Cb_Cg_Nd2_diangle = 180.0 + self.Ca_Cb_Cg_Od1_diangle

        # Add common/shared amino acid geometric properties. 
        # NOTE: This should always be last, as the computation of the # of dihedral angles relies 
        # on the number of 'dihedral' in the variable names. AGAIN, THIS IS HACKY.
        super(AsnGeometry, self).__init__()


class GluGeometry(AminoAcidGeometry):
    """Glutamic acid geometry."""

    def __init__(self):
        self.residue_name = "E"
        self.full_name = "Glu"

        # Amino acid specific bonds, angles, and dihedrals
        self.N_Ca_C_angle = 111.1703
        self.Ca_C_O_angle = 120.511
        self.N_Ca_C_O_diangle = 120.0

        self.Ca_Cb_length = 1.52
        self.C_Ca_Cb_angle = 109.5
        self.N_C_Ca_Cb_diangle = 122.8702

        self.Cb_Cg_length = 1.52
        self.Ca_Cb_Cg_angle = 113.82
        self.N_Ca_Cb_Cg_diangle = -63.8

        self.Cg_Cd_length = 1.52
        self.Cb_Cg_Cd_angle = 113.31
        self.Ca_Cb_Cg_Cd_diangle = -179.8

        self.Cd_Oe1_length = 1.25
        self.Cg_Cd_Oe1_angle = 119.02
        self.Cb_Cg_Cd_Oe1_diangle = -6.2

        self.Cd_Oe2_length = 1.25
        self.Cg_Cd_Oe2_angle = 118.08    
        self.Cb_Cg_Cd_Oe2_diangle = 180.0 + self.Cb_Cg_Cd_Oe1_diangle

        # Add common/shared amino acid geometric properties. 
        # NOTE: This should always be last, as the computation of the # of dihedral angles relies 
        # on the number of 'dihedral' in the variable names. AGAIN, THIS IS HACKY.
        super(GluGeometry, self).__init__()


class GlnGeometry(AminoAcidGeometry):
    """Glutamine geometry."""

    def __init__(self):
        self.residue_name = "Q"
        self.full_name = "Gln"

        # Amino acid specific bonds, angles, and dihedrals
        self.N_Ca_C_angle = 111.0849
        self.Ca_C_O_angle = 120.5029
        self.N_Ca_C_O_diangle = 120.0

        self.Ca_Cb_length = 1.52
        self.C_Ca_Cb_angle = 109.5
        self.N_C_Ca_Cb_diangle = 122.8134

        self.Cb_Cg_length = 1.52
        self.Ca_Cb_Cg_angle = 113.75
        self.N_Ca_Cb_Cg_diangle = -60.2

        self.Cg_Cd_length = 1.52
        self.Cb_Cg_Cd_angle = 112.78
        self.Ca_Cb_Cg_Cd_diangle = -69.6

        self.Cd_Oe1_length = 1.24
        self.Cg_Cd_Oe1_angle = 120.86
        self.Cb_Cg_Cd_Oe1_diangle = -50.5

        self.Cd_Ne2_length = 1.33
        self.Cg_Cd_Ne2_angle = 116.50
        self.Cb_Cg_Cd_Ne2_diangle = 180 + self.Cb_Cg_Cd_Oe1_diangle

        # Add common/shared amino acid geometric properties. 
        # NOTE: This should always be last, as the computation of the # of dihedral angles relies 
        # on the number of 'dihedral' in the variable names. AGAIN, THIS IS HACKY.
        super(GlnGeometry, self).__init__()


class MetGeometry(AminoAcidGeometry):
    """Methionine geometry."""

    def __init__(self):
        self.residue_name = "M"
        self.full_name = "Met"

        # Amino acid specific bonds, angles, and dihedrals
        self.N_Ca_C_angle = 110.9416
        self.Ca_C_O_angle = 120.4816
        self.N_Ca_C_O_diangle = 120.0

        self.Ca_Cb_length = 1.52
        self.C_Ca_Cb_angle = 109.5
        self.N_C_Ca_Cb_diangle = 122.6733

        self.Cb_Cg_length = 1.52
        self.Ca_Cb_Cg_angle =  113.68
        self.N_Ca_Cb_Cg_diangle = -64.4

        self.Cg_Sd_length = 1.81
        self.Cb_Cg_Sd_angle = 112.69
        self.Ca_Cb_Cg_Sd_diangle = -179.6

        self.Sd_Ce_length = 1.79
        self.Cg_Sd_Ce_angle = 100.61
        self.Cb_Cg_Sd_Ce_diangle = 70.1

        # Add common/shared amino acid geometric properties. 
        # NOTE: This should always be last, as the computation of the # of dihedral angles relies 
        # on the number of 'dihedral' in the variable names. AGAIN, THIS IS HACKY.
        super(MetGeometry, self).__init__()


class HisGeometry(AminoAcidGeometry):
    """Histidine geometry."""

    def __init__(self):
        self.residue_name = "H"
        self.full_name = "His"

        # Amino acid specific bonds, angles, and dihedrals
        self.N_Ca_C_angle = 111.0859
        self.Ca_C_O_angle = 120.4732
        self.N_Ca_C_O_diangle = 120.0

        self.Ca_Cb_length = 1.52
        self.C_Ca_Cb_angle = 109.5
        self.N_C_Ca_Cb_diangle = 122.6711

        self.Cb_Cg_length = 1.49
        self.Ca_Cb_Cg_angle = 113.74
        self.N_Ca_Cb_Cg_diangle = -63.2
        
        self.Cg_Nd1_length = 1.38
        self.Cb_Cg_Nd1_angle = 122.85
        self.Ca_Cb_Cg_Nd1_diangle = -75.7          

        self.Cg_Cd2_length = 1.35
        self.Cb_Cg_Cd2_angle = 130.61
        self.Ca_Cb_Cg_Cd2_diangle = 180.0 + self.Ca_Cb_Cg_Nd1_diangle

        self.Nd1_Ce1_length = 1.32
        self.Cg_Nd1_Ce1_angle = 108.5
        self.Cb_Cg_Nd1_Ce1_diangle = 180.0

        self.Cd2_Ne2_length = 1.35
        self.Cg_Cd2_Ne2_angle = 108.5
        self.Cb_Cg_Cd2_Ne2_diangle = 180.0

        # Add common/shared amino acid geometric properties. 
        # NOTE: This should always be last, as the computation of the # of dihedral angles relies 
        # on the number of 'dihedral' in the variable names. AGAIN, THIS IS HACKY.
        super(HisGeometry, self).__init__()


class ProGeometry(AminoAcidGeometry):
    """Proline geometry."""

    def __init__(self):
        self.residue_name = "P"
        self.full_name = "Pro"

        # Amino acid specific bonds, angles, and dihedrals
        self.N_Ca_C_angle=112.7499
        self.Ca_C_O_angle=120.2945
        self.N_Ca_C_O_diangle=-45.0

        self.Ca_Cb_length=1.52
        self.C_Ca_Cb_angle=109.5
        self.N_C_Ca_Cb_diangle=115.2975

        self.Cb_Cg_length=1.49
        self.Ca_Cb_Cg_angle=104.21
        self.N_Ca_Cb_Cg_diangle=29.6

        self.Cg_Cd_length=1.50
        self.Cb_Cg_Cd_angle=105.03
        self.Ca_Cb_Cg_Cd_diangle=-34.8

        # Add common/shared amino acid geometric properties. 
        # NOTE: This should always be last, as the computation of the # of dihedral angles relies 
        # on the number of 'dihedral' in the variable names. AGAIN, THIS IS HACKY.
        super(ProGeometry, self).__init__()


class PheGeometry(AminoAcidGeometry):
    """Phenylalanine geometry."""

    def __init__(self):
        self.residue_name = "F"
        self.full_name = "Phe"

        # Amino acid specific bonds, angles, and dihedrals
        self.N_Ca_C_angle=110.7528
        self.Ca_C_O_angle=120.5316
        self.N_Ca_C_O_diangle=120.0

        self.Ca_Cb_length=1.52
        self.C_Ca_Cb_angle=109.5
        self.N_C_Ca_Cb_diangle=122.6054

        self.Cb_Cg_length=1.50
        self.Ca_Cb_Cg_angle=113.85
        self.N_Ca_Cb_Cg_diangle=-64.7

        self.Cg_Cd1_length=1.39
        self.Cb_Cg_Cd1_angle=120.0
        self.Ca_Cb_Cg_Cd1_diangle=93.3

        self.Cg_Cd2_length=1.39
        self.Cb_Cg_Cd2_angle=120.0
        self.Ca_Cb_Cg_Cd2_diangle=self.Ca_Cb_Cg_Cd1_diangle - 180.0

        self.Cd1_Ce1_length=1.39
        self.Cg_Cd1_Ce1_angle=120.0
        self.Cb_Cg_Cd1_Ce1_diangle=180.0

        self.Cd2_Ce2_length=1.39
        self.Cg_Cd2_Ce2_angle=120.0
        self.Cb_Cg_Cd2_Ce2_diangle=180.0

        self.Ce1_Cz_length=1.39
        self.Cd1_Ce1_Cz_angle=120.0
        self.Cg_Cd1_Ce1_Cz_diangle=0.0

        # Add common/shared amino acid geometric properties. 
        # NOTE: This should always be last, as the computation of the # of dihedral angles relies 
        # on the number of 'dihedral' in the variable names. AGAIN, THIS IS HACKY.
        super(PheGeometry, self).__init__()


class TyrGeometry(AminoAcidGeometry):
    """Tyrosine geometry."""

    def __init__(self):
        self.residue_name = "Y"
        self.full_name = "Tyr"

        # Amino acid specific bonds, angles, and dihedrals
        self.N_Ca_C_angle=110.9288
        self.Ca_C_O_angle=120.5434
        self.N_Ca_C_O_diangle=120.0

        self.Ca_Cb_length=1.52
        self.C_Ca_Cb_angle=109.5
        self.N_C_Ca_Cb_diangle=122.6023

        self.Cb_Cg_length=1.51
        self.Ca_Cb_Cg_angle= 113.8
        self.N_Ca_Cb_Cg_diangle=-64.3

        self.Cg_Cd1_length=1.39
        self.Cb_Cg_Cd1_angle=120.98
        self.Ca_Cb_Cg_Cd1_diangle=93.1

        self.Cg_Cd2_length=1.39
        self.Cb_Cg_Cd2_angle=120.82
        self.Ca_Cb_Cg_Cd2_diangle=self.Ca_Cb_Cg_Cd1_diangle + 180.0

        self.Cd1_Ce1_length=1.39
        self.Cg_Cd1_Ce1_angle=120.0
        self.Cb_Cg_Cd1_Ce1_diangle=180.0

        self.Cd2_Ce2_length=1.39
        self.Cg_Cd2_Ce2_angle=120.0
        self.Cb_Cg_Cd2_Ce2_diangle=180.0

        self.Ce1_Cz_length=1.39
        self.Cd1_Ce1_Cz_angle=120.0
        self.Cg_Cd1_Ce1_Cz_diangle=0.0

        self.Cz_OH_length=1.39
        self.Ce1_Cz_OH_angle=119.78
        self.Cd1_Ce1_Cz_OH_diangle=180.0

        # Add common/shared amino acid geometric properties. 
        # NOTE: This should always be last, as the computation of the # of dihedral angles relies 
        # on the number of 'dihedral' in the variable names. AGAIN, THIS IS HACKY.
        super(TyrGeometry, self).__init__()


class TrpGeometry(AminoAcidGeometry):
    """Tryptophan geometry."""

    def __init__(self):
        self.residue_name = "W"
        self.full_name = "Trp"

        # Amino acid specific bonds, angles, and dihedrals
        self.N_Ca_C_angle=110.8914
        self.Ca_C_O_angle=120.5117
        self.N_Ca_C_O_diangle=120.0

        self.Ca_Cb_length=1.52
        self.C_Ca_Cb_angle=109.5
        self.N_C_Ca_Cb_diangle=122.6112

        self.Cb_Cg_length=1.50
        self.Ca_Cb_Cg_angle=114.10
        self.N_Ca_Cb_Cg_diangle=-66.4

        self.Cg_Cd1_length=1.37
        self.Cb_Cg_Cd1_angle=127.07
        self.Ca_Cb_Cg_Cd1_diangle=96.3

        self.Cg_Cd2_length=1.43
        self.Cb_Cg_Cd2_angle=126.66
        self.Ca_Cb_Cg_Cd2_diangle=self.Ca_Cb_Cg_Cd1_diangle - 180.0

        self.Cd1_Ne1_length=1.38
        self.Cg_Cd1_Ne1_angle=108.5
        self.Cb_Cg_Cd1_Ne1_diangle=180.0

        self.Cd2_Ce2_length=1.40
        self.Cg_Cd2_Ce2_angle=108.5
        self.Cb_Cg_Cd2_Ce2_diangle=180.0

        self.Cd2_Ce3_length=1.40
        self.Cg_Cd2_Ce3_angle=133.83
        self.Cb_Cg_Cd2_Ce3_diangle=0.0

        self.Ce2_Cz2_length=1.40
        self.Cd2_Ce2_Cz2_angle=120.0
        self.Cg_Cd2_Ce2_Cz2_diangle=180.0

        self.Ce3_Cz3_length=1.40
        self.Cd2_Ce3_Cz3_angle=120.0
        self.Cg_Cd2_Ce3_Cz3_diangle=180.0

        self.Cz2_CH2_length=1.40
        self.Ce2_Cz2_CH2_angle=120.0
        self.Cd2_Ce2_Cz2_CH2_diangle=0.0

        # Add common/shared amino acid geometric properties. 
        # NOTE: This should always be last, as the computation of the # of dihedral angles relies 
        # on the number of 'dihedral' in the variable names. AGAIN, THIS IS HACKY.
        super(TrpGeometry, self).__init__()


AA_geometries = {
    "G": GlyGeometry(),
    "A": AlaGeometry(),
    "S": SerGeometry(),
    "C": CysGeometry(),
    "V": ValGeometry(), 
    "I": IleGeometry(),
    "L": LeuGeometry(),
    "T": ThrGeometry(),
    "R": ArgGeometry(),
    "K": LysGeometry(),
    "D": AspGeometry(),
    "E": GluGeometry(),
    "N": AsnGeometry(),
    "Q": GlnGeometry(),
    "M": MetGeometry(),
    "H": HisGeometry(),
    "P": ProGeometry(),
    "F": PheGeometry(),
    "Y": TyrGeometry(),
    "W": TrpGeometry()
}
