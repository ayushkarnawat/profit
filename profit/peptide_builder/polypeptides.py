"""Polypetide representation(s) and conversion(s)."""

aa1 = list("ACDEFGHIKLMNPQRSTVWY")
aa3 = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS",
       "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL",
       "TRP", "TYR"]

three_to_one = {aa_name: aa1[idx] for idx, aa_name in enumerate(aa3)}
one_to_three = {value: key for key, value in three_to_one.items()}


def is_aa(residue: str) -> bool:
    """Determine if valid amino acid."""
    residue = residue.upper()
    return residue in aa1 or residue in aa3