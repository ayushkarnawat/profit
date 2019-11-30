from profit.peptide_builder.polypeptides import one_to_three, is_aa

AA1_VOCAB = {
    "U": -1, # unknown
    "X": 0,  # pad
    "A": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "Y": 20
}

AA3_VOCAB = {
    "UNK": -1, # unknown
    "XXX": 0,  # pad
}
AA3_VOCAB.update({one_to_three.get(k):v for k,v in AA1_VOCAB.items() if is_aa(k)})

FLIPPED_AA1 = {v:k for k,v in AA1_VOCAB.items()}
FLIPPED_AA3 = {v:k for k,v in AA3_VOCAB.items()}