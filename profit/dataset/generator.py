"""Generates variants of amino acid sequence."""

import itertools

import numpy as np
import pandas as pd

from profit.utils.io_utils import maybe_create_dir


def gen(n: int) -> np.ndarray:
    """Generate all possible combinations of `n` length AA sequence.

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
    acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
             'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    combs = list(itertools.product(acids, repeat=n))
    return np.array(["".join(comb) for comb in combs])


def generate(save_path: str, n: int = 4) -> None:
    """Saves all possible combinations of `n` length amino acid sequence.

    TODO: Add var that allows you to remove the certain variants. Useful
    if we want to generate a dataset that doesn't include variants from
    the training dataset.

    Params:
    -------
    save_path: str
        Path to store generated data.

    n: int, default=4
        Length of the sequence.
    """
    seqs = gen(n)
    n_total = seqs[:].shape[0]
    df = pd.DataFrame({
        'ID': np.arange(n_total),
        'Variants': seqs,
        'Fitness': ['' for _ in range(n_total)]
    })

    # Save dataset to file. Checks if intended filepath is available.
    save_path = maybe_create_dir(save_path)
    df.to_csv(save_path, sep=',', header=True, index=False, columns=['ID', 'Variants', 'Fitness'])
    print('Saved generated variants to {0:s}'.format(save_path))


if __name__ == "__main__":
    from tqdm import tqdm
    from profit.dataset.preprocessors.lstm_preprocessor import LSTMPreprocessor
    from profit.utils.data_utils.serializers import LMDBSerializer

    # Generate (all) variants
    seqs = gen(n=4)
    pos = [39, 40, 41, 54]
    template = list("MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE")

    # Compute features
    pp = LSTMPreprocessor(vocab="iupac1")
    features = []
    for seq in tqdm(seqs, total=len(seqs)):
        assert len(seq) == len(pos)
        for idx, aa in zip(pos, seq):
            template[idx-1] = aa
        features.append(pp.get_input_feats(template))
    features = np.array(features)

    # Save dataset
    LMDBSerializer.save(features, path="data/3gb1/processed/lstm_fitness/variants.mdb")

    # Compute features (aka embeddings) for those cases
    # generate('data/raw/variants{}.csv'.format(20**n), n=n)
    