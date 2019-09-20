"""Definitions and properties of amino acids for Peptide Library Package"""

# Natural Amino Acids
aminos = {
    'Glycine': {
        'Code': 'Gly',
        'Formula': 'C2H5NO2',
        'Letter': 'G',
        'MolWeight': '75.07',
        'SMILES': 'NCC(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': False
    },
    'L-Alanine': {
        'Code': 'Ala',
        'Formula': 'C3H7NO2',
        'Letter': 'A',
        'MolWeight': '89.09',
        'SMILES': 'N[C@@]([H])(C)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': False
    },
    'L-Arginine': {
        'Code': 'Arg',
        'Formula': 'C6H14N4O2',
        'Letter': 'R',
        'MolWeight': '174.20',
        'SMILES': 'N[C@@]([H])(CCCNC(=N)N)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': False
    },
    'L-Asparagine': {
        'Code': 'Asn',
        'Formula': 'C4H8N2O3',
        'Letter': 'N',
        'MolWeight': '132.12',
        'SMILES': 'N[C@@]([H])(CC(=O)N)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': False
    },
    'L-Aspartic_Acid': {
        'Code': 'Asp',
        'Formula': 'C4H7NO4',
        'Letter': 'D',
        'MolWeight': '133.10',
        'SMILES': 'N[C@@]([H])(CC(=O)O)C(=O)O',
        'cterm': 'N[C@@]([H])(CC*(=O))C(=O)O',
        'disulphide': False,
        'ester': False,
        'nterm': False
    },
    'L-Cysteine': {
        'Code': 'Cys',
        'Formula': 'C3H7NO2S',
        'Letter': 'C',
        'MolWeight': '121.16',
        'SMILES': 'N[C@@]([H])(CS)C(=O)O',
        'cterm': False,
        'disulphide': 'N[C@@]([H])(CS*)C(=O)O',
        'ester': False,
        'nterm': False
    },
    'L-Glutamic_Acid': {
        'Code': 'Glu',
        'Formula': 'C5H9NO4',
        'Letter': 'E',
        'MolWeight': '147.13',
        'SMILES': 'N[C@@]([H])(CCC(=O)O)C(=O)O',
        'cterm': 'N[C@@]([H])(CCC*(=O))C(=O)O',
        'disulphide': False,
        'ester': False,
        'nterm': False
    },
    'L-Glutamine': {
        'Code': 'Gln',
        'Formula': 'C5H10N2O3',
        'Letter': 'Q',
        'MolWeight': '146.15',
        'SMILES': 'N[C@@]([H])(CCC(=O)N)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': False
    },
    'L-Histidine': {
        'Code': 'His',
        'Formula': 'C6H9N3O2',
        'Letter': 'H',
        'MolWeight': '155.16',
        'SMILES': 'N[C@@]([H])(CC1=CN=C-N1)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': False
    },
    'L-Isoleucine': {
        'Code': 'Ile',
        'Formula': 'C6H13NO2',
        'Letter': 'I',
        'MolWeight': '131.18',
        'SMILES': 'N[C@@]([H])([C@]([H])(CC)C)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': False
    },
    'L-Leucine': {
        'Code': 'Leu',
        'Formula': 'C6H13NO2',
        'Letter': 'L',
        'MolWeight': '131.18',
        'SMILES': 'N[C@@]([H])(CC(C)C)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': False
    },
    'L-Lysine': {
        'Code': 'Lys',
        'Formula': 'C6H12N2O2',
        'Letter': 'K',
        'MolWeight': '146.19',
        'SMILES': 'N[C@@]([H])(CCCCN)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': 'N[C@@]([H])(CCCCN*)C(=O)O'
    },
    'L-Methionine': {
        'Code': 'Met',
        'Formula': 'C5H11NO2S',
        'Letter': 'M',
        'MolWeight': '149.21',
        'SMILES': 'N[C@@]([H])(CCSC)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': False
    },
    'L-Phenylalanine': {
        'Code': 'Phe',
        'Formula': 'C9H11NO2',
        'Letter': 'F',
        'MolWeight': '165.19',
        'SMILES': 'N[C@@]([H])(Cc1ccccc1)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': False
    },
    'L-Proline': {
        'Code': 'Pro',
        'Formula': 'C5H9NO2',
        'Letter': 'P',
        'MolWeight': '115.13',
        'SMILES': 'N1[C@@]([H])(CCC1)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': False
    },
    'L-Serine': {
        'Code': 'Ser',
        'Formula': 'C3H7NO2',
        'Letter': 'S',
        'MolWeight': '105.09',
        'SMILES': 'N[C@@]([H])(CO)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': 'N[C@@]([H])(CO*)C(=O)O',
        'nterm': False
    },
    'L-Threonine': {
        'Code': 'Thr',
        'Formula': 'C4H9NO3',
        'Letter': 'T',
        'MolWeight': '119.12',
        'SMILES': 'N[C@@]([H])([C@]([H])(O)C)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': 'N[C@@]([H])([C@]([H])(O*)C)C(=O)O',
        'nterm': False
    },
    'L-Tryptophan': {
        'Code': 'Trp',
        'Formula': 'C11H12N2O2',
        'Letter': 'W',
        'MolWeight': '204.23',
        'SMILES': 'N[C@@]([H])(CC(=CN2)C1=C2C=CC=C1)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': False
    },
    'L-Tyrosine': {
        'Code': 'Tyr',
        'Formula': 'C9H11NO3',
        'Letter': 'Y',
        'MolWeight': '181.19',
        'SMILES': 'N[C@@]([H])(Cc1ccc(O)cc1)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': 'N[C@@]([H])(Cc1ccc(O*)cc1)C(=O)O',
        'nterm': False
    },
    'L-Valine': {
        'Code': 'Val',
        'Formula': 'C5H11NO2',
        'Letter': 'V',
        'MolWeight': '117.15',
        'SMILES': 'N[C@@]([H])(C(C)C)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': False
    }
}

# Opposite chirality aminos 
d_aminos = {
    'D-Alanine': {
        'Code': 'ala',
        'Formula': 'C3H7NO2',
        'Letter': 'a',
        'MolWeight': '89.09',
        'SMILES': 'N[C@]([H])(C)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': False
    },
    'D-Arginine': {
        'Code': 'arg',
        'Formula': 'C6H14N4O2',
        'Letter': 'r',
        'MolWeight': '174.20',
        'SMILES': 'N[C@]([H])(CCCNC(=N)N)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': False
    },
    'D-Asparagine': {
        'Code': 'asn',
        'Formula': 'C4H8N2O3',
        'Letter': 'n',
        'MolWeight': '132.12',
        'SMILES': 'N[C@]([H])(CC(=O)N)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': False
    },
    'D-Aspartic_Acid': {
        'Code': 'asp',
        'Formula': 'C4H7NO4',
        'Letter': 'd',
        'MolWeight': '133.10',
        'SMILES': 'N[C@]([H])(CC(=O)O)C(=O)O',
        'cterm': 'N[C@]([H])(CC*(=O))C(=O)O',
        'disulphide': False,
        'ester': False,
        'nterm': False
    },
    'D-Cysteine': {
        'Code': 'cys',
        'Formula': 'C3H7NO2S',
        'Letter': 'c',
        'MolWeight': '121.16',
        'SMILES': 'N[C@]([H])(CS)C(=O)O',
        'cterm': False,
        'disulphide': 'N[C@]([H])(CS*)C(=O)O',
        'ester': False,
        'nterm': False
    },
    'D-Glutamic_Acid': {
        'Code': 'glu',
        'Formula': 'C5H9NO4',
        'Letter': 'e',
        'MolWeight': '147.13',
        'SMILES': 'N[C@]([H])(CCC(=O)O)C(=O)O',
        'cterm': 'N[C@]([H])(CCC*(=O))C(=O)O',
        'disulphide': False,
        'ester': False,
        'nterm': False
    },
    'D-Glutamine': {
        'Code': 'gln',
        'Formula': 'C5H10N2O3',
        'Letter': 'q',
        'MolWeight': '146.15',
        'SMILES': 'N[C@]([H])(CCC(=O)N)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': False
    },
    'D-Histidine': {
        'Code': 'his',
        'Formula': 'C6H9N3O2',
        'Letter': 'h',
        'MolWeight': '155.16',
        'SMILES': 'N[C@]([H])(CC1=CN=C-N1)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': False
    },
    'D-Isoleucine': {
        'Code': 'ile',
        'Formula': 'C6H13NO2',
        'Letter': 'i',
        'MolWeight': '131.18',
        'SMILES': 'N[C@]([H])([C@@]([H])(CC)C)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': False
    },
    'D-Leucine': {
        'Code': 'leu',
        'Formula': 'C6H13NO2',
        'Letter': 'l',
        'MolWeight': '131.18',
        'SMILES': 'N[C@]([H])(CC(C)C)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': False
    },
    'D-Lysine': {
        'Code': 'lys',
        'Formula': 'C6H12N2O2',
        'Letter': 'k',
        'MolWeight': '146.19',
        'SMILES': 'N[C@]([H])(CCCCN)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': 'N[C@]([H])(CCCCN*)C(=O)O'
    },
    'D-Methionine': {
        'Code': 'met',
        'Formula': 'C5H11NO2S',
        'Letter': 'm',
        'MolWeight': '149.21',
        'SMILES': 'N[C@]([H])(CCSC)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': False
    },
    'D-Phenylalanine': {
        'Code': 'phe',
        'Formula': 'C9H11NO2',
        'Letter': 'f',
        'MolWeight': '165.19',
        'SMILES': 'N[C@]([H])(Cc1ccccc1)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': False
    },
    'D-Proline': {
        'Code': 'pro',
        'Formula': 'C5H9NO2',
        'Letter': 'p',
        'MolWeight': '115.13',
        'SMILES': 'N1[C@]([H])(CCC1)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': False
    },
    'D-Serine': {
        'Code': 'ser',
        'Formula': 'C3H7NO2',
        'Letter': 's',
        'MolWeight': '105.09',
        'SMILES': 'N[C@]([H])(CO)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': 'N[C@]([H])(CO*)C(=O)O',
        'nterm': False
    },
    'D-Tryptophan': {
        'Code': 'trp',
        'Formula': 'C11H12N2O2',
        'Letter': 'w',
        'MolWeight': '204.23',
        'SMILES': 'N[C@]([H])(CC(=CN2)C1=C2C=CC=C1)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': False
    },
    'D-Tyrosine': {
        'Code': 'tyr',
        'Formula': 'C9H11NO3',
        'Letter': 'y',
        'MolWeight': '181.19',
        'SMILES': 'N[C@]([H])(Cc1ccc(O)cc1)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': 'N[C@]([H])(Cc1ccc(O*)cc1)C(=O)O',
        'nterm': False
    },
    'D-Valine': {
        'Code': 'val',
        'Formula': 'C5H11NO2',
        'Letter': 'v',
        'MolWeight': '117.15',
        'SMILES': 'N[C@]([H])(C(C)C)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': False
    },
    'D-Threonine': {
        'Code': 'thr',
        'Formula': 'C4H9NO3',
        'Letter': 't',
        'MolWeight': '119.12',
        'SMILES': 'N[C@]([H])([C@@]([H])(O)C)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': 'N[C@]([H])([C@@]([H])(O*)C)C(=O)O',
        'nterm': False
    }
}

# Opposite chirality aminos 
special_aminos = {
    'L-Diaminopropionic_Acid': {
        'Code': 'Dap',
        'Formula': 'C3H8N2O2',
        'Letter': 'J',
        'MolWeight': '104.12',
        'SMILES': 'N[C@@]([H])(CN)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': 'N[C@@]([H])(CN*)C(=O)O'
    },
    'L-Ornithine': {
        'Code': 'Orn',
        'Formula': 'C5H12N2O2',
        'Letter': 'O',
        'MolWeight': '132.16',
        'SMILES': 'N[C@@]([H])(CCCN)C(=O)O',
        'cterm': False,
        'disulphide': False,
        'ester': False,
        'nterm': 'N[C@@]([H])(CCCN*)C(=O)O'
    }
}
