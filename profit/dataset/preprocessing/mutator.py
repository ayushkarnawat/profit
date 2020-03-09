import os
import json
from typing import Dict, List, Optional, Union

import numpy as np
from rdkit.Chem import rdchem, rdForceFieldHelpers, rdmolfiles
from rdkit.Geometry.rdGeometry import Point3D

import pymol
from pymol import __main__, cmd
from pymol.wizard.mutagenesis import Mutagenesis

from profit.peptide_builder.polypeptides import aa1, aa3, three_to_one, one_to_three, is_aa
from profit.utils.io_utils import maybe_create_dir, DownloadError


def _get_conformer(mol: rdchem.Mol,
                   conformer: str = "min",
                   algo: str = "MMFF") -> rdchem.Mol:
    """Get molecule conformer from PDB file based on parameters
    provided.

    Params:
    -------
    mol: rdkit.Chem.rdchem.Mol
        Molecule of interest, ideally with mutiple conformers.

    conformer: str, optional, default="min"
        Which conformer to select for 3D coordinates. If "min" (or "max"),
        then the conformer with the min (or max) energy is selected. If
        "first" or "last", then the first or last conformer is selected.
        If "avg", then the average position of all the conformers are
        averaged.

    algo: str, optional, default="MMFF"
        Which force field algorithm to optimize the coordinates with.
        Read rdkit description to determine which one is best suited
        for your application.

    Returns:
    --------
    mol: rdkit.Chem.rdchem.Mol
        Molecule with conformer of interest.
    """
    forcefields = {
        "MMFF": rdForceFieldHelpers.MMFFOptimizeMoleculeConfs,
        "UFF":  rdForceFieldHelpers.UFFOptimizeMoleculeConfs
    }

    if conformer == "min":
         # Get idx of lowest energy conformation
        idx = np.argmin(forcefields[algo](mol, maxIters=0), axis=0)[1]
        conf = mol.GetConformers()[idx]
    elif conformer == "max":
        # Get idx of highest energy conformation
        idx = np.argmax(forcefields[algo](mol, maxIters=0), axis=0)[1]
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
            conf.SetAtomPosition(atom_idx, Point3D(atom_coords[0], \
                atom_coords[1], atom_coords[2]))
    else:
        available_confs = ["min", "max", "first", "last", "avg"]
        raise ValueError(f"Cannot get `{conformer}` conformer. Choose from the "
                         f"following {available_confs} conformer(s).")

    # Save conformer, with the position specified
    mol.RemoveAllConformers()
    mol.AddConformer(conf)
    return mol


class PDBMutator(object):
    """Insilico point mutation of peptide residue(s).

    Mutates residues with other residue(s) defined by the user at the
    specified positions in the chain.

    After a mutation is made at a certain position, the conformation of
    the mutant sidechain is optimized by some molecular dynamics (MD)
    simulation using a certain forcefield, i.e. Merck's molecular
    (accuracy) or coarse-grained (speed).

    Params:
    -------
    fmt: str, default="tertiary"
        Primary or tertiary structure. If "primary" structure, then the
        mutator simply replaces the peptide string with another peptide
        string. If "tertiary", then atoms associated with the residue
        (at the specified XYZ location) get replaced by atoms in the
        other residue, usually with default rotamer configurations.

    rootdir: str, default="data/tmp"
        Base directory where all cached data is stored.

    cache: bool, default=False
        If "True", the mutated files are saved after they are computed
        and returned. If "False" (default), they are deleted.
    """

    def __init__(self,
                 fmt: str = "tertiary",
                 rootdir: str = "data/tmp",
                 cache: bool = False) -> None:
        self.fmt = fmt
        self.cache = cache
        self.rootdir = rootdir

        # Launch PyMol quietly (no msg) with no GUI
        __main__.pymol_argv = ["pymol", "-qc"]
        pymol.finish_launching()


    def mutate(self, pdbid: str,
               replace_with: Dict[int, Optional[str]]) -> Union[List[str], rdchem.Mol]:
        """Modify amino acid residues at the defined positions.

        If the locations indexes exceed the amount in data, then they
        will be ignored and a warning will be announced.

        Params:
        -------
        pdbid: str
            PDB ID associated with the structure.

        replace_with: dict
            The index location(s) within the full protein to replace
            certain residue(s) with. If a residue associated with a
            index location is None, then the modified residue is chosen
            randomly. If a certain index exceeds the number of available
            residues in the protein, then those enteries are simply
            ignored and the user is notified.

        Returns:
        --------
        protein: list of str or rdkit.Chem.rdchem.Mol
            Modified protein with residues. If fmt="primary", then list
            of string (peptide names) is returned. If fmt="tertiary",
            then 3D molecule structure is returned.
        """
        # Load PDB structure (download, if necessary)
        pdb_dir = maybe_create_dir(os.path.join(self.rootdir, pdbid))
        pdb_file = os.path.join(pdb_dir, f"{pdbid}.pdb")
        if not os.path.exists(pdb_file):
            is_successful = cmd.fetch(pdbid, name=pdbid, state=1, type="pdb", path=pdb_dir)
            if is_successful == -1:
                raise DownloadError(f"Unable to download '{pdbid}'.")
        else:
            cmd.load(pdb_file, object=pdbid, state=1, format="pdb")

        # Get all residue names, see: https://pymolwiki.org/index.php/List_Selection
        resnames_dict = {"names": []}
        cmd.iterate("(name ca)", "names.append(resn)", space=resnames_dict)
        residue_names = resnames_dict["names"]
        num_residues = len(residue_names)

        # Cleanup idxs: remove indicies that exceed number of available residues
        nonvalid_idxs = [idx for idx in replace_with.keys() if idx > num_residues]
        for idx in nonvalid_idxs:
            print(f"OutOfRange: Removing idx {idx} (only {num_residues} residues).")
            replace_with.pop(idx)

        # Randomly choose an amino acid (AA) to replace residue, if None is provided.
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
                raise ValueError(f"Invalid residue '{residue}'. Choose one from "
                                 f"the following {aa1+aa3}.")

        # Determine save filepath name
        modified_res_str = ":".join([f"{k}{three_to_one.get(v)}"
                                     for k, v in replace_with.items()])
        filename = f"{self.fmt}_{modified_res_str}"
        filename += ".pdb" if self.fmt == "tertiary" else ".json"
        save_filepath = os.path.join(self.rootdir, pdbid, filename)

        # Replace primary structure, i.e. residue names (str)
        if self.fmt == "primary":
            # Load data from cache, if it exists
            protein = None
            if os.path.exists(save_filepath):
                with open(save_filepath) as json_file:
                    protein = json.load(json_file)
            if protein is None:
                for idx, residue in replace_with.items():
                    residue_names[idx-1] = residue # since PDB starts with idx of 1
                protein = [three_to_one.get(name) for name in residue_names]

                # Save sequence temporarily
                _ = maybe_create_dir(save_filepath)
                with open(save_filepath, "w") as outfile:
                    json.dump(protein, outfile)
        # Replace tertiary structure, i.e. residue's 3D coordinates
        elif self.fmt == "tertiary":
            if not os.path.exists(save_filepath):
                # Split states so that we can optimize only on specific state(s).
                # NOTE: Might be useful to choose lowest energy state to mutate,
                # OR mutate rotamers for all positions, then choose one with
                # lowest energy.
                cmd.split_states(object=pdbid)

                # Delete all other objects other than one we want to mutate
                # NOTE: For now, keep only first object. This might change
                # depending on which state needs to be kept.
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

                # Save PDB temporarily
                _ = maybe_create_dir(save_filepath)
                cmd.save(save_filepath, selection=pdbid, format="pdb")
                cmd.delete("all") # remove all objects, clears workspace

            # Load + choose model/structure with lowest energy
            # NOTE: If sanitize=True, the function checks if Mol has the correct
            # hybridization/valance structure (aka is it chemically reasonable).
            # When converting from the PDB block, this sometimes results in
            # improper parsing. Instead, for now, we just check if the Mol is
            # syntactically valid (i.e. all rings/branches closed, no illegal
            # atom types, etc).
            protein = rdmolfiles.MolFromPDBFile(save_filepath, sanitize=False, removeHs=False)
            if protein.GetNumConformers() > 1:
                protein = _get_conformer(protein, conformer="min", algo="MMFF")
        else:
            raise NotImplementedError

        # Remove file, if not needed
        if not self.cache:
            os.remove(save_filepath)

        return protein
