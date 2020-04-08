"""Substitution matrices.

Substitution matrices can be scored on a variety of factors. For
example, BLOSUM matrices score amino acid substitutions by their
appearances throughout evolution, as they compare the frequencies of
different mutations in similar blocks of sequences [1]. Others employ
chemical similarity and neighbourhood selectivity [2]. In fact, when the
stability effects of mutations are evaluated, the frequency of an amino
acid substitution in nature may not be the most important factor.
Alternatively, mixtures of substitution models have also been shown to
work well [3].

Regardless of the scoring scheme used, each element in the matrix
denotes the substitution probabilty between the 20 naturally occuring
amino acids [4]. Note that the rows and columns are both indexed by the
order denoted in :class:`peptide_builder.polypedetides.aa1`.

TODO: Change ordering to be based off the order defined in aa1? As of
now, the ordering is correct, but if the underlying vocab ordering
changes, then things will break (i.e. the substitution probabilty will
not be correct).

References:
-----------
[1] Henikoff,S. and Henikoff,J.G. (1992) Amino acid substitution
    matrices from protein blocks. Proc. Natl. Acad. Sci. USA, 89,
    10915–10919.

[2] Tomii,K. and Kanehisa,M. (1996) Analysis of amino acid indices and
    mutation matrices for sequence comparison and structure prediction
    of proteins. Protein Eng. Des. Select., 9, 27–36.

[3] Cichonska,A. et al. (2017) Computational-experimental approach to
    drug-target interaction mapping: a case study on kinase inhibitors.
    PLoS Comput. Biol., 13, e1005678.

[4] Shen et al. Introduction to the Peptide Binding Problem of Com-
    putational Immunology: New Results. Foundations of Computational
    Mathematics, 14(5):951–984, Oct 2014. ISSN 1615-3375. DOI:
    10.1007/s10208-013-9173-9.
"""

import numpy as np


BLOSUM62 = np.array(
    [[3.9029, 0.6127, 0.5883, 0.5446, 0.8680, 0.7568, 0.7413, 1.0569, 0.5694, 0.6325,
      0.6019, 0.7754, 0.7232, 0.4649, 0.7541, 1.4721, 0.9844, 0.4165, 0.5426, 0.9365],
     [0.6127, 6.6656, 0.8586, 0.5732, 0.3089, 1.4058, 0.9608, 0.4500, 0.9170, 0.3548,
      0.4739, 2.0768, 0.6226, 0.3807, 0.4815, 0.7672, 0.6778, 0.3951, 0.5560, 0.4201],
     [0.5883, 0.8586, 7.0941, 1.5539, 0.3978, 1.0006, 0.9113, 0.8637, 1.2220, 0.3279,
      0.3100, 0.9398, 0.4745, 0.3543, 0.4999, 1.2315, 0.9842, 0.2778, 0.4860, 0.3690],
     [0.5446, 0.5732, 1.5539, 7.3979, 0.3015, 0.8971, 1.6878, 0.6343, 0.6786, 0.3390,
      0.2866, 0.7841, 0.3465, 0.2990, 0.5987, 0.9135, 0.6948, 0.2321, 0.3457, 0.3365],
     [0.8680, 0.3089, 0.3978, 0.3015, 19.5766, 0.3658, 0.2859, 0.4204, 0.3550, 0.6535,
      0.6423, 0.3491, 0.6114, 0.4390, 0.3796, 0.7384, 0.7406, 0.4500, 0.4342, 0.7558],
     [0.7568, 1.4058, 1.0006, 0.8971, 0.3658, 6.2444, 1.9017, 0.5386, 1.1680, 0.3829,
      0.4773, 1.5543, 0.8643, 0.3340, 0.6413, 0.9656, 0.7913, 0.5094, 0.6111, 0.4668],
     [0.7413, 0.9608, 0.9113, 1.6878, 0.2859, 1.9017, 5.4695, 0.4813, 0.9600, 0.3305,
      0.3729, 1.3083, 0.5003, 0.3307, 0.6792, 0.9504, 0.7414, 0.3743, 0.4965, 0.4289],
     [1.0569, 0.4500, 0.8637, 0.6343, 0.4204, 0.5386, 0.4813, 6.8763, 0.4930, 0.2750,
      0.2845, 0.5889, 0.3955, 0.3406, 0.4774, 0.9036, 0.5793, 0.4217, 0.3487, 0.3370],
     [0.5694, 0.9170, 1.2220, 0.6786, 0.3550, 1.1680, 0.9600, 0.4930, 13.5060, 0.3263,
      0.3807, 0.7789, 0.5841, 0.6520, 0.4729, 0.7367, 0.5575, 0.4441, 1.7979, 0.3394],
     [0.6325, 0.3548, 0.3279, 0.3390, 0.6535, 0.3829, 0.3305, 0.2750, 0.3263, 3.9979,
      1.6944, 0.3964, 1.4777, 0.9458, 0.3847, 0.4432, 0.7798, 0.4089, 0.6304, 2.4175],
     [0.6019, 0.4739, 0.3100, 0.2866, 0.6423, 0.4773, 0.3729, 0.2845, 0.3807, 1.6944,
      3.7966, 0.4283, 1.9943, 1.1546, 0.3711, 0.4289, 0.6603, 0.5680, 0.6921, 1.3142],
     [0.7754, 2.0768, 0.9398, 0.7841, 0.3491, 1.5543, 1.3083, 0.5889, 0.7789, 0.3964,
      0.4283, 4.7643, 0.6253, 0.3440, 0.7038, 0.9319, 0.7929, 0.3589, 0.5322, 0.4565],
     [0.7232, 0.6226, 0.4745, 0.3465, 0.6114, 0.8643, 0.5003, 0.3955, 0.5841, 1.4777,
      1.9943, 0.6253, 6.4815, 1.0044, 0.4239, 0.5986, 0.7938, 0.6103, 0.7084, 1.2689],
     [0.4649, 0.3807, 0.3543, 0.2990, 0.4390, 0.3340, 0.3307, 0.3406, 0.6520, 0.9458,
      1.1546, 0.3440, 1.0044, 8.1288, 0.2874, 0.4400, 0.4817, 1.3744, 2.7694, 0.7451],
     [0.7541, 0.4815, 0.4999, 0.5987, 0.3796, 0.6413, 0.6792, 0.4774, 0.4729, 0.3847,
      0.3711, 0.7038, 0.4239, 0.2874, 12.8375, 0.7555, 0.6889, 0.2818, 0.3635, 0.4431],
     [1.4721, 0.7672, 1.2315, 0.9135, 0.7384, 0.9656, 0.9504, 0.9036, 0.7367, 0.4432,
      0.4289, 0.9319, 0.5986, 0.4400, 0.7555, 3.8428, 1.6139, 0.3853, 0.5575, 0.5652],
     [0.9844, 0.6778, 0.9842, 0.6948, 0.7406, 0.7913, 0.7414, 0.5793, 0.5575, 0.7798,
      0.6603, 0.7929, 0.7938, 0.4817, 0.6889, 1.6139, 4.8321, 0.4309, 0.5732, 0.9809],
     [0.4165, 0.3951, 0.2778, 0.2321, 0.4500, 0.5094, 0.3743, 0.4217, 0.4441, 0.4089,
      0.5680, 0.3589, 0.6103, 1.3744, 0.2818, 0.3853, 0.4309, 38.1078, 2.1098, 0.3745],
     [0.5426, 0.5560, 0.4860, 0.3457, 0.4342, 0.6111, 0.4965, 0.3487, 1.7979, 0.6304,
      0.6921, 0.5322, 0.7084, 2.7694, 0.3635, 0.5575, 0.5732, 2.1098, 9.8322, 0.6580],
     [0.9365, 0.4201, 0.3690, 0.3365, 0.7558, 0.4668, 0.4289, 0.3370, 0.3394, 2.4175,
      1.3142, 0.4565, 1.2689, 0.7451, 0.4431, 0.5652, 0.9809, 0.3745, 0.6580, 3.6922]]
)
