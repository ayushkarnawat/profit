"""Generates variants of amino acid sequence."""

import logging
import itertools
from typing import Optional

import numpy as np
import pandas as pd

from profit.utils.io import maybe_create_dir

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s  %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def gen(n: Optional[int]=4) -> np.ndarray:
    """Generate all possible combinations of `n` length amino acid sequence.

    Params:
    -------
    n: int
        Length of the sequence. 

    Returns:
    --------
    seqs: np.ndarray, shape=(20^n,)
        All possible permutations of `n` length amino acids sequence. 
    
    Example:
    --------
    >>> combs = gen(n=2)
    ['AA', 'AR', 'AN', 'AD', ..., 'VV']
    """
    acids = ['A', 'R', 'N', 'D', 'C', 
             'Q', 'E', 'G', 'H', 'I', 
             'L', 'K', 'M', 'F', 'P', 
             'S', 'T', 'W', 'Y', 'V']
    combs = list(itertools.product(acids, repeat=n))
    return np.array(["".join(comb) for comb in combs])


def generate(save_path: str, n: Optional[int]=4) -> None:
    """Saves all possible combinations of `n` length amino acid sequence. 

    TODO: Add var that allows you to remove the certain variants. Useful to 
    generate a dataset that doesn't include variants in your training dataset.
    
    Params:
    -------
    save_path: str
        Path to store generated data. Directories will be created if necessary.

    n: int, optional, default=4
        Length of the sequence.
    """
    seqs = gen(n)
    n_total = seqs.shape[0]
    df = pd.DataFrame({'ID': np.arange(n_total), 
                       'Variants': seqs, 
                       'Fitness': ['' for _ in range(n_total)] # TODO: should it be 0. or ''?
                      })

    # Save dataset to file. Checks if intended filepath is available.
    save_path = maybe_create_dir(save_path)
    df.to_csv(save_path, sep=',', header=True, index=False, columns=['ID', 'Variants', 'Fitness'])
    logger.info('Saved generated variants to {0:s}'.format(save_path))


if __name__ == "__main__":
    n=4
    generate('data/raw/variants{}.csv'.format(20**n), n=n)
    