"""Different vocab types."""

from collections import OrderedDict
from profit.peptide_builder.polypeptides import aa1, aa3


# See: https://www.dnastar.com/megalign_help/index.html#!Documents/iupaccodesforaminoacids.htm
_IUPAC_AA_CODES = OrderedDict([
    ("ALA", "A"),
    ("ARG", "R"),
    ("ASN", "N"),
    ("ASP", "D"),
    # ("ASX", "B"), # Aspartic Acid or Asparagine
    ("CYS", "C"),
    ("GLN", "Q"),
    ("GLU", "E"),
    # ("GLX", "Z"), # Glutamic Acid or Glutamine
    ("GLY", "G"),
    ("HIS", "H"),
    ("ILE", "I"),
    ("LEU", "L"),
    ("LYS", "K"),
    ("MET", "M"),
    ("PHE", "F"),
    ("PRO", "P"),
    ("PYL", "O"), # synthetic; genetically encoded
    ("SEC", "U"), # synthetic; genetically encoded
    ("SER", "S"),
    ("THR", "T"),
    ("TRP", "W"),
    ("TYR", "Y"),
    ("VAL", "V"),
    # ("XAA", "X"), # Unspecified or unknown
    # ("XLE", "J"), # Leucine or Isoleucine
])


_BASE_STUB = OrderedDict([
    ("<pad>", 0),   # padding token
    ("<unk>", 1),   # unknown token
])

_BERT_STUB = _BASE_STUB.copy()
_BERT_STUB.update([
    ("<mask>", 2),  # hidden (masked) token
    ("<cls>", 3),   # classification token (beginning of sentence)
    ("<sep>", 4),   # seperation token (end of sentence)
])

IUPAC_AA1 = _BERT_STUB.copy()
IUPAC_AA1.update({
    aa1: len(IUPAC_AA1)+i for i, aa1 in enumerate(_IUPAC_AA_CODES.values())
})

IUPAC_AA3 = _BERT_STUB.copy()
IUPAC_AA3.update({
    aa3: len(IUPAC_AA3)+i for i, aa3 in enumerate(_IUPAC_AA_CODES.keys())
})

NATURAL_AA1 = _BASE_STUB.copy()
NATURAL_AA1.update({aa: len(NATURAL_AA1)+i for i, aa in enumerate(aa1)})

NATURAL_AA3 = _BASE_STUB.copy()
NATURAL_AA3.update({aa: len(NATURAL_AA3)+i for i, aa in enumerate(aa3)})

# We do not require this vocab to have the base stub (aka <pad>, <unk>) since we
# assume that there will be no variable length sequences and no unknown tokens.
ONLY_AA20 = OrderedDict({aa: i for i, aa in enumerate(aa1)})


# NOTE: We must define all vocabs here rather than within the parent directory's
# __init__ because we want to use them in tokenizers
VOCABS = {
    "iupac1": IUPAC_AA1,
    "iupac3": IUPAC_AA3,
    "aa1": NATURAL_AA1,
    "aa3": NATURAL_AA3,
    "aa20": ONLY_AA20,
}
