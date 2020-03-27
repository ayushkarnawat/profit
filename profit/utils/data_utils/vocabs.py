"""Different vocab types."""

from collections import OrderedDict

# See: https://www.dnastar.com/megalign_help/index.html#!Documents/iupaccodesforaminoacids.htm
IUPAC_AA_CODES = OrderedDict([
    ("ALA", "A"),
    ("ASX", "B"),
    ("CYS", "C"),
    ("ASP", "D"),
    ("GLU", "E"),
    ("PHE", "F"),
    ("GLY", "G"),
    ("HIS", "H"),
    ("ILE", "I"),
    ("XLE", "J"),
    ("LYS", "K"),
    ("LEU", "L"),
    ("MET", "M"),
    ("ASN", "N"),
    ("PRO", "P"),
    ("GLN", "Q"),
    ("ARG", "R"),
    ("SER", "S"),
    ("THR", "T"),
    ("SEC", "U"),
    ("VAL", "V"),
    ("TRP", "W"),
    ("XAA", "X"),
    ("TYR", "Y"),
    ("GLX", "Z")
])

IUPAC_AA1_VOCAB = OrderedDict([
    ("<pad>", 0),   # padding token
    ("<mask>", 1),  # hidden (masked) token
    ("<cls>", 2),   # classification token (beginning of sentence)
    ("<sep>", 3),   # seperation token (end of sentence)
    ("<unk>", 4),   # unknown token
    ("A", 5),
    ("B", 6),
    ("C", 7),
    ("D", 8),
    ("E", 9),
    ("F", 10),
    ("G", 11),
    ("H", 12),
    ("I", 13),
    ("J", 14),
    ("K", 15),
    ("L", 16),
    ("M", 17),
    ("N", 18),
    ("O", 19),
    ("P", 20),
    ("Q", 21),
    ("R", 22),
    ("S", 23),
    ("T", 24),
    ("U", 25),
    ("V", 26),
    ("W", 27),
    ("X", 28),
    ("Y", 29),
    ("Z", 30)
])

IUPAC_AA3_VOCAB = OrderedDict([
    ("<pad>", 0),   # padding token
    ("<mask>", 1),  # hidden (masked) token
    ("<cls>", 2),   # classification token (beginning of sentence)
    ("<sep>", 3),   # seperation token (end of sentence)
    ("<unk>", 4),   # unknown token
])
IUPAC_AA3_VOCAB.update({k:IUPAC_AA1_VOCAB.get(v) for k, v in IUPAC_AA_CODES.items()})


AA20_VOCAB = OrderedDict([
    ("<pad>", 0),   # padding token
    ("<unk>", 1),   # unknown token
    ("A", 2),
    ("C", 3),
    ("D", 4),
    ("E", 5),
    ("F", 6),
    ("G", 7),
    ("H", 8),
    ("I", 9),
    ("K", 10),
    ("L", 11),
    ("M", 12),
    ("N", 13),
    ("P", 14),
    ("Q", 15),
    ("R", 16),
    ("S", 17),
    ("T", 18),
    ("V", 19),
    ("W", 20),
    ("Y", 21),
])
