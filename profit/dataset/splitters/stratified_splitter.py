"""Stratified splitting. Adapted from https://git.io/JvriE."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from profit.dataset.splitters.base_splitter import BaseSplitter


def _approximate_mode(class_counts: np.ndarray, n_draws: int) -> np.ndarray:
    """Computes approximate mode of a multivariate hypergeometric.

    Draws `n_draws` number of samples from the population given by
    class_counts. NOTE: The sampled distribution output should be very
    similar to the initial distribution of the class_counts (i.e. should
    not be off by more than one).

    Params:
    -------
    class_counts: np.ndarray of ints
        Number of samples (aka population) per class.

    n_draws: int
        Number of draws (samples to draw) from the overall population.

    Returns:
    --------
    sampled_classes: np.ndarray
        Number of samples drawn from each class.
        NOTE: np.sum(sampled_classes) == n_draws

    Examples:
    ---------
    >>> _approximate_mode(class_counts=np.array([4,2]), n_draws=3)
    array([2,1])
    >>> _approximate_mode(class_counts=np.array([2, 2, 2, 1]), n_draws=2)
    array([0,1,1,0])
    """
    n_class = len(class_counts)
    continuous = class_counts * n_draws / class_counts.sum()
    floored = np.floor(continuous)
    assert n_draws // n_class == floored.sum() // n_class
    n_remainder = int(n_draws - floored.sum())
    remainder = continuous - floored
    inds = np.argsort(remainder)[::-1]
    inds = inds[:n_remainder]
    floored[inds] += 1
    assert n_draws == floored.sum()
    return floored.astype(np.int)


class StratifiedSplitter(BaseSplitter):
    """Class for performing stratified data splits.

    For classification problems, the stratified split will contain the
    approximately the same amount of samples from each class within the
    train, validation, and test sets. Similarly, for regression-based
    tasks (continuous labels), the labels are binned into quantiles and
    sampled from each quantile, ensuring that each set has a balanced
    representation of the full data distribution.

    NOTE: Stratified sampling is most appropiate when (a) the sample size
    `n` is limited and (b) small sub-groups or 'strata' maybe over or
    under represented in each split set. Having these two conditions in
    the dataset can skew the metrics, if the data is randomly split.
    """

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
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

        seed = kwargs.get('seed', None)
        labels = kwargs.get('labels', None)
        n_bins = kwargs.get('n_bins', 10)
        task_type = kwargs.get('task_type', 'auto')
        if task_type not in ['classification', 'regression', 'auto']:
            raise ValueError(f"{task_type} is invalid. Please use "
                             "'classification', 'regression', or 'auto'")

        rng = np.random.RandomState(seed)

        # Parse labels, if necessary
        if isinstance(labels, list):
            labels = np.array(labels)
        elif labels is None: # if the labels are combined with the data matrix X
            # Obtain col index containing labels, useful for multitask training
            label_axis = kwargs.get('label_axis', -1)
            labels = dataset[..., label_axis]

        # Checks if the label values are all int type. Ensures that the label
        # axis is correctly typed to determine downstream training task (i.e.
        # classification or regression).
        if labels.dtype.kind == 'f' or labels.dtype.kind == 'i':
            if np.all(np.mod(labels, 1) == 0):
                labels = labels.astype(int)
        else:
            raise ValueError(f"Invalid label type {labels.dtype.kind}. Please "
                             "ensure your labels are ints or floats.")

        if task_type == 'auto':
            if labels.dtype.kind == 'i': # int
                task_type = 'classification'
            elif labels.dtype.kind == 'f': # float
                task_type = 'regression'

        if task_type == 'classification':
            classes, labels = np.unique(labels, return_inverse=True)
        elif task_type == 'regression':
            # Quantile cut the dataset into the max number of sufficient bins
            # while dropping duplicate bin edges. In this case, the number of
            # classes is at most n_bins (aka n_classes <= n_bins).
            labels = pd.qcut(labels, n_bins, labels=False, duplicates='drop')
            classes = np.unique(labels)

        n_classes = classes.shape[0]
        n_total_val = int(frac_valid * len(dataset))
        n_total_test = int(frac_test * len(dataset))

        class_counts = np.bincount(labels)
        class_idxs = np.split(np.argsort(labels, kind='mergesort'), np.cumsum(class_counts)[:-1])

        # Stratify samples from distribution for both validation and test sets.
        # Remainder is the distribution of the training set. NOTE: The percentage
        # of each class/bin should be equally represented in all 3 sets.
        n_val_samples = _approximate_mode(class_counts, n_total_val)
        class_counts -= n_val_samples # remove val samples from total before sampling for test set
        n_test_samples = _approximate_mode(class_counts, n_total_test)

        train_idxs, val_idxs, test_idxs = [], [], []
        for i in range(n_classes):
            n_val = n_val_samples[i]
            n_test = n_test_samples[i]

            # Shuffle idx for each binned class
            shuffled = rng.permutation(len(class_idxs[i]))
            class_shuffled_idx = class_idxs[i][shuffled]

            class_train_idx = class_shuffled_idx[n_val+n_test:]
            class_val_idx = class_shuffled_idx[:n_val]
            class_test_idx = class_shuffled_idx[n_val: n_val+n_test]

            train_idxs.extend(class_train_idx)
            val_idxs.extend(class_val_idx)
            test_idxs.extend(class_test_idx)

        assert n_total_val == len(val_idxs)
        assert n_total_test == len(test_idxs)
        return np.array(train_idxs), np.array(val_idxs), np.array(test_idxs)


    def train_valid_test_split(self,
                               dataset: np.ndarray,
                               labels: Optional[np.ndarray] = None,
                               frac_train: float = 0.8,
                               frac_valid: float = 0.1,
                               frac_test: float = 0.1,
                               n_bins: int = 10,
                               return_idxs: bool = True,
                               seed: Optional[int] = None,
                               task_type: str = 'auto',
                               **kwargs: Dict[str, Any]) -> Tuple[np.ndarray, ...]:
        """
        Generate indicies to split dataset into train, val, and test sets.

        Params:
        -------
        dataset: np.ndarray, size=(n,m)
            Dataset where `n` is the number of examples, and `m` is the
            number of features.

        labels: np.ndarray or None, optional, default=None
            Target labels of size=(n,). If `None`, function assumes that
            the last column of the dataset is the labels.

        frac_train: float, default=0.8
            Fraction of data to be used in training.

        frac_valid: float, default=0.1
            Fraction of data to be used in validation.

        frac_test: float, default=0.1
            Fraction of data to be used in testing.

        n_bins: int, default=10
            The number of bins to subset labels if task_type is continous.

        return_idxs: bool, default=True
            If `True`, this function returns only indicies. If `False`,
            returns the splitted dataset.

        seed: int or None, optional, default=None
            Random state seed.

        task_type: bool, optional, default='auto'
            Type of training task. If `auto`, automatically determines
            task between classification and regression based off labels
            data type.

        Returns:
        --------
        splitted_data: tuple of np.ndarrays
            The splitted data (train, val, and test) or their indicies.
        """
        return super(StratifiedSplitter, self).train_valid_test_split(
            dataset, frac_train, frac_valid, frac_test, return_idxs,
            labels=labels, n_bins=n_bins, seed=seed, task_type=task_type,
            **kwargs)


    def train_valid_split(self,
                          dataset: np.ndarray,
                          labels: Optional[np.ndarray] = None,
                          frac_train: float = 0.8,
                          frac_valid: float = 0.2,
                          n_bins: int = 10,
                          return_idxs: bool = True,
                          seed: Optional[int] = None,
                          task_type: str = 'auto',
                          **kwargs: Dict[str, Any]) -> Tuple[np.ndarray, ...]:
        """
        Generate indicies to split dataset into train and val sets.

        Params:
        -------
        dataset: np.ndarray, size=(n,m)
            Dataset where `n` is the number of examples, and `m` is the
            number of features.

        labels: np.ndarray or None, optional, default=None
            Target labels of size=(n,). If `None`, function assumes that
            the last column of the dataset is the labels.

        frac_train: float, default=0.8
            Fraction of data to be used in training.

        frac_valid: float, default=0.2
            Fraction of data to be used in validation.

        n_bins: int, default=10
            The number of bins to subset labels if task_type is continous.

        return_idxs: bool, default=True
            If `True`, this function returns only indicies. If `False`,
            returns the splitted dataset.

        seed: int or None, optional, default=None
            Random state seed.

        task_type: bool, optional, default='auto'
            Type of training task. If `auto`, automatically determines
            task between classification and regression based off labels
            data type.

        Returns:
        --------
        splitted_data: tuple of np.ndarrays
            The splitted data (train and val) or their indicies.
        """
        return super(StratifiedSplitter, self).train_valid_split(
            dataset, frac_train, frac_valid, return_idxs, labels=labels,
            n_bins=n_bins, seed=seed, task_type=task_type, **kwargs)
