"""Amino acid name(s) and conversion(s)."""


aa1 = list("ARNDCQEGHILKMFPSTWYV")
aa3 = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
       "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
       "TYR", "VAL"]

three_to_one = {aa_name: aa1[idx] for idx, aa_name in enumerate(aa3)}
one_to_three = {value: key for key, value in three_to_one.items()}


def is_aa(residue: str) -> bool:
    """Determine if valid amino acid."""
    residue = residue.upper()
    return residue in aa1 or residue in aa3
