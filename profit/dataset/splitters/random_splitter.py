"""Random splitting. Adapted from https://git.io/Jvri0."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from profit.dataset.splitters.base_splitter import BaseSplitter


class RandomSplitter(BaseSplitter):
    """Class for random splitting of data."""

    def _split(self,
               dataset: np.ndarray,
               frac_train: float = 0.8,
               frac_valid: float = 0.1,
               frac_test: float = 0.1,
               **kwargs: Dict[str, Any]) -> Tuple[np.ndarray, ...]:
        """
        Split dataset into their respective sets.

        Params:
        -------
        dataset: np.ndarray, size=(n,m)
            Dataset where `n` is the number of examples, and `m` is the
            number of features.

        frac_train: float, default=0.8
            Fraction of data to be used in training.

        frac_valid: float, default=0.1
            Fraction of data to be used in validation.

        frac_test: float, default=0.1
            Fraction of data to be used in testing.

        Returns:
        --------
        splitted_data: tuple of np.ndarrays
            The train, val, and test set indicies, respectively.
        """
        # Test inputs
        seed = kwargs.get('seed', None)
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

        if seed is not None:
            # Ensure same permutation is generated everytime
            shuffled = np.random.RandomState(seed).permutation(len(dataset))
        else:
            shuffled = np.random.permutation(len(dataset))
        train_size = int(len(dataset) * frac_train)
        val_size = int(len(dataset) * frac_valid)
        return (shuffled[:train_size], shuffled[train_size:train_size + val_size],
                shuffled[train_size + val_size:])


    def train_valid_test_split(self,
                               dataset: np.ndarray,
                               frac_train: float = 0.8,
                               frac_valid: float = 0.1,
                               frac_test: float = 0.1,
                               return_idxs: bool = True,
                               seed: Optional[int] = None,
                               **kwargs: Dict[str, Any]) -> Tuple[np.ndarray, ...]:
        """
        Generate indicies to split dataset into train, val, and test sets.

        Params:
        -------
        dataset: np.ndarray, size=(n,m)
            Dataset where `n` is the number of examples, and `m` is the
            number of features.

        frac_train: float, default=0.8
            Fraction of data to be used in training.

        frac_valid: float, default=0.1
            Fraction of data to be used in validation.

        frac_test: float, default=0.1
            Fraction of data to be used in testing.

        return_idxs: bool, default=True
            If `True`, this function returns only indicies. If `False`,
            returns the splitted dataset.

        seed: int or None, optional, default=None
            Random state seed.

        Returns:
        --------
        splitted_data: tuple of np.ndarrays
            The splitted data (train, val, and test) or their indicies.
        """
        return super(RandomSplitter, self)\
            .train_valid_test_split(dataset, frac_train, frac_valid, frac_test,
                                    return_idxs, seed=seed, **kwargs)


    def train_valid_split(self,
                          dataset: np.ndarray,
                          frac_train: float = 0.8,
                          frac_valid: float = 0.2,
                          return_idxs: bool = True,
                          seed: Optional[int] = None,
                          **kwargs: Dict[str, Any]) -> Tuple[np.ndarray, ...]:
        """
        Generate indicies to split dataset into train and val sets.

        Params:
        -------
        dataset: np.ndarray, size=(n,m)
            Dataset where `n` is the number of examples, and `m` is the
            number of features.

        frac_train: float, default=0.8
            Fraction of data to be used in training.

        frac_valid: float, default=0.2
            Fraction of data to be used in validation.

        return_idxs: bool, default=True
            If `True`, this function returns only indicies. If `False`,
            returns the splitted dataset.

        seed: int or None, optional, default=None
            Random state seed.

        Returns:
        --------
        splitted_data: tuple of np.ndarrays
            The splitted data (train and val) or their indicies.
        """
        return super(RandomSplitter, self)\
            .train_valid_split(dataset, frac_train, frac_valid, return_idxs,
                               seed=seed, **kwargs)
