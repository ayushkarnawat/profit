import os
from typing import Dict, List, Optional, Union

import numpy as np
from rdkit.Chem import rdchem, rdForceFieldHelpers, rdmolfiles
from rdkit.Geometry.rdGeometry import Point3D

from profit.utils.io import maybe_create_dir, DownloadError


aa1 = "ACDEFGHIKLMNPQRSTVWY"
aa3 = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS",
       "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL",
       "TRP", "TYR"]

three_to_one = {aa_name: aa1[idx] for idx, aa_name in enumerate(aa3)}
one_to_three = {value: key for key, value in three_to_one.items()}


def is_aa(residue: str) -> bool:
    """Determine if valid amino acid."""
    residue = residue.upper()
    return residue in list(aa1) or residue in aa3


def _get_conformer(mol: rdchem.Mol, conformer: str="min", algo: str="MMFF") -> rdchem.Mol:
    """Get molecule conformer from PDB file based on parameters provided.
    
    Params:
    -------
    mol: rdkit.Chem.rdchem.Mol
        Molecule of interest, ideally with mutiple conformers.

    conformer: str, optional, default="min"
        Which conformer to select for 3D coordinates. If "min" (or "max"), then the conformer with 
        the min (or max) energy is selected. If "first" or "last", then the first or last conformer 
        is selected. If "avg", then the average position of all the conformers are averaged.

    algo: str, optional, default='MMFF'
        Which force field algorithm to optimize the coordinates with. Read rdkit description to 
        determine which one is best suited for your application.

    Returns:
    --------
    mol: rdkit.Chem.rdchem.Mol
        Molecule with conformer of interest.
    """
    ff = {
        "MMFF": rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, maxIters=0), 
        "UFF":  rdForceFieldHelpers.UFFOptimizeMoleculeConfs(mol, maxIters=0) 
    }

    if conformer == "min":
        idx = np.argmin(ff[algo], axis=0)[1] # get idx of lowest energy conformation
        conf = mol.GetConformers()[idx]
    elif conformer == "max":
        idx = np.argmax(ff[algo], axis=0)[1] # get idx of highest energy conformation
        conf = mol.GetConformers()[idx]
    elif conformer == "first":
        conf = mol.GetConformer(0)
    elif conformer == "last":
        conf = mol.GetConformer(mol.GetNumConformers()-1)
    elif conformer == "avg":
        allpos = [conf.GetPositions() for conf in mol.GetConformers()]
        avgpos = np.average(allpos, axis=0)
        # Set avg position as new position for all atoms
        conf = mol.GetConformer(0)
        for atom_idx in range(conf.GetNumAtoms()):
            atom_coords = avgpos[atom_idx]
            conf.SetAtomPosition(atom_idx, Point3D(atom_coords[0], atom_coords[1], atom_coords[2]))
    else:
        available_confs = ['min', 'max', 'first', 'last', 'avg']
        raise ValueError("Cannot get `{}` conformer. Choose from the following "
                         "{} conformer(s).".format(conformer, available_confs))

    # Save conformer, with the position specified
    mol.RemoveAllConformers()
    mol.AddConformer(conf)
    return mol


class PDBMutator(object):
    """Insilico point mutation of peptide residue(s) with other peptide residue(s) defined by the 
    user at the specified positions in the chain. 

    After a mutation is made at a certain position, the conformation of the mutant sidechain is 
    optimized by some molecular dynamics (MD) simulation using a certain forcefield, i.e. Merck's 
    molecular (accuracy) or coarse-grained (speed).

    Params:
    -------
    fmt: str, optional, default="tertiary"
        Primary or tertiary structure. If "primary" structure, then the mutator simply replaces 
        the peptide string with another peptide string. If "tertiary", then the atoms associated 
        with the residue (at the specified XYZ location) get replaced by atoms in the other 
        residue, usually with default rotamer configurations.

    remove_tmp_file: bool, optional, default=True
        If 'True', the mutated PDB files are removed after they are computed and returned. 
    """
    
    def __init__(self, fmt: str="tertiary", remove_tmp_file: bool=True):
        self.fmt = fmt
        self.remove_tmp_file = remove_tmp_file

    
    def modify_residues(self, pdbid: str, replace_with: Dict[int, Optional[str]]) \
                        -> Union[List[str], rdchem.Mol]:
        """Modify amino acid residues at the defined positions.
        
        If the locations indexes exceed the amount in data, then they will be ignored and a 
        warning will be announced.

        Params:
        -------
        pdbid: str
            PDB ID associated with the structure.

        replace_with: dict
            The index location(s) within the full protein to replace certain residue(s) with. 
            If a residue associated with a index location is None, then the modified residue is 
            chosen randomly. If a certain index exceeds the number of available residues in the 
            protein, then those enteries are simply ignored and the user is notified.
        
        Returns:
        --------
        peptide: list of str or rdkit.Chem.rdchem.Mol
            Modified peptide with residues. If fmt='primary', then list of string (peptide names) 
            is returned. If fmt='tertiary', then 3D molecule structure is returned.
        """
        # Launch PyMol quietly (no msg) with no GUI
        import __main__
        import pymol
        from pymol import cmd
        from pymol.wizard.mutagenesis import Mutagenesis
        __main__.pymol_argv = ['pymol', '-qc']
        pymol.finish_launching()

        # Load PDB structure (download, if necessary)
        pdb_dir = maybe_create_dir("data/pdb/")
        is_successful = cmd.fetch(pdbid, name=pdbid, state=1, type='pdb', path=pdb_dir)
        if is_successful == -1:
            raise DownloadError("Unable to download '{0:s}'.".format(pdbid))

        # Get all residue names, see: https://pymolwiki.org/index.php/List_Selection
        resnames_dict = {'names': []}
        cmd.iterate("(name ca)", "names.append(resn)", space=resnames_dict)
        residue_names = resnames_dict['names']
        num_residues = len(residue_names)

        # Cleanup idxs: remove indicies that exceed number of available residues
        nonvalid_idxs = [idx for idx in replace_with.keys() if idx > num_residues]
        for idx in nonvalid_idxs:
            print("Removing idx {0:d} (out of range). There are only {1:d} "
                  "residue(s).".format(idx, num_residues))
            replace_with.pop(idx)

        # Randomly choose an amino acid (AA) to replace a residue, if None is provided.
        # Additionally, format string such that it is a valid 3 letter amino acid. 
        for idx, residue in replace_with.items():
            if residue is None:
                replace_with[idx] = np.random.choice(aa3)
            elif is_aa(residue):
                residue = residue.upper()
                if len(residue) == 1:
                    replace_with[idx] = one_to_three.get(residue)
                elif len(residue) == 3:
                    replace_with[idx] = residue
            else:
                raise ValueError("Invalid residue {}. Choose one from the " 
                                 "following {}.".format(residue, aa3))

        # Replace primary structure, i.e. residue names (str)
        if self.fmt == "primary":
            for idx, residue in replace_with.items():
                residue_names[idx-1] = residue # since PDB starts with idx of 1
            return [three_to_one.get(name) for name in residue_names]
        # Replace tertiary structure, i.e. residue's 3D coordinates
        elif self.fmt == "tertiary":
            # Split states so that we can optimize only on specific state(s).
            # NOTE: It might be useful to choose lowest energy state to mutate, OR
            # mutate rotamers for all positions, then choose one with lowest energy.
            cmd.split_states(object=pdbid)

            # Delete all other objects other than one we want to mutate
            # NOTE: For now, keep only first object. This might change depending on which state needs to be kept.
            objs = cmd.get_object_list() # aka states
            keep_objs = [pdbid + "_0001"]
            for obj in objs:
                if obj not in keep_objs:
                    cmd.delete(obj)
            assert keep_objs == cmd.get_object_list()

            # Mutate residues
            cmd.wizard("mutagenesis")
            wizard: Mutagenesis = cmd.get_wizard()
            for idx, res in replace_with.items():
                selection = "{0:s}//A/{1:d}/".format(keep_objs[0], idx)
                wizard.do_select(selection) # select which residue index to replace
                wizard.set_mode(res)        # choose name of residue to replace with
                wizard.do_state(1)          # select rotamer with least strain (aka conflicts w/ other atoms)
                wizard.apply()              # apply point mutation
            cmd.set_wizard(None) # close wizard

            # Save PDB temporarily and extract for later use
            mutated_res_ids = "".join(list(map(lambda resname: three_to_one.get(resname), replace_with.values())))
            save_path = "data/tmp/{0:s}_mutated_{1:s}.pdb".format(pdbid, mutated_res_ids)
            _ = maybe_create_dir(save_path)
            cmd.save(save_path, selection=pdbid, format="pdb")

            # Choose model/structure with lowest energy
            # NOTE: If sanitize=True, the function checks for correct hybridization/valance structure.
            # This sometimes results in the improper parsing of Mol instance. For now, we ignore this. 
            mol = rdmolfiles.MolFromPDBFile(save_path, sanitize=False, removeHs=False)
            if mol.GetNumConformers() > 1:
                mol = _get_conformer(mol, conformer="min", algo="MMFF")

            # Remove file, if not needed
            if self.remove_tmp_file:
                os.remove(save_path)

            return mol
