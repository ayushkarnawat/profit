import numpy as np
import pandas as pd

from profit.dataset.splitter.base_splitter import BaseSplitter
from typing import Any, Dict, Optional, Tuple, Union


def _approximate_mode(class_counts: np.ndarray, n_draws: int) -> np.ndarray:
    """Computes approximate mode of a multivariate hypergeometric. 

    Draws `n_draws` number of samples from the population given by class_counts. 
    NOTE: The sampled distribution output should be very similar to the initial 
    distribution of the class_counts (i.e. should not be off by more than one).

    Params:
    -------
    class_counts: np.ndarray of ints
        Number of samples (aka population) per class.

    n_draws: int
        Number of draws (samples to draw) from the overall population.

    Returns:
    --------
    sampled_classes: np.ndarray
        Number of samples drawn from each class. NOTE: np.sum(sampled_classes) == n_draws

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
    """Class for performing stratified data splits. Adapted from https://git.io/JeGt8.
    
    For classification problems, the stratified split will contain the approximately the same 
    amount of samples from each class within the train, validation, and test sets. Similarly, 
    for regressions tasks (continuous tasks), the labels are binned into quantiles and sampled 
    from each quantile, ensuring that each set has a balanced representation of the full data 
    distribution.

    NOTE: Stratified sampling is most appropiate when (a) the sample size `n` is limited and (b) 
    small sub-groups or 'strata' maybe over or under represented in each split set. Having these 
    two conditions in the dataset can skew the metrics, if the data is randomly split.
    """

    def _split(self,
               dataset: np.ndarray, 
               frac_train: Optional[float]=0.8, 
               frac_val: Optional[float]=0.1, 
               frac_test: Optional[float]=0.1, 
               **kwargs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split dataset into their respective sets.

        Params:
        -------
        dataset: np.ndarray, size=(n,m)
            Dataset where `n` is the number of examples, and `m` is number of features.

        frac_train: float, optional, default=0.8
            Fraction of data to be used in training. 

        frac_val: float, optional, default=0.1 
            Fraction of data to be used in validation.

        frac_test: float, optional, default=0.1
            Fraction of data to be used in testing.

        Returns:
        --------
        splitted_data: Tuple[np.ndarray, np.ndarray, np.ndarray]
            The splitted data into their train, validation, and test sets, respectively.
        """
        # Test inputs
        np.testing.assert_almost_equal(frac_train + frac_val + frac_test, 1.0)

        seed = kwargs.get('seed', None)
        labels = kwargs.get('labels', None)
        n_bins = kwargs.get('n_bins', 10) 
        task_type = kwargs.get('task_type', 'auto')
        if task_type not in ['classification', 'regression', 'auto']:
            raise ValueError("{0:s} is invalid. Please use 'classification',"
                             "'regression', or 'auto'".format(task_type))

        rng = np.random.RandomState(seed)

        # Parse labels, if necessary
        if isinstance(labels, list):
            labels = np.array(labels)
        elif labels is None: # if the labels are combined with the data matrix X
            label_axis = kwargs.get('label_axis', -1) # column index of labels, useful if training for multitask 
            labels = dataset[:, label_axis]

        # Checks if the label values are all int type. Ensures that the label axis is correctly 
        # typed to determine downstream training task (i.e. classification or regression). 
        if labels.dtype.kind == 'f' or labels.dtype.kind == 'i':
            if np.all(np.mod(labels, 1) == 0):
                labels = labels.astype(int)
        else:
            raise ValueError("Invalid label type {0:s}. Please ensure your labels are ints or "
                             "floats.".format(labels.dtype.kind))

        if task_type == 'auto':
            if labels.dtype.kind == 'i': # int
                task_type = 'classification'
            elif labels.dtype.kind == 'f': # float
                task_type = 'regression'

        if task_type == 'classification':
            classes, labels = np.unique(labels, return_inverse=True)
        elif task_type == 'regression':
            classes = np.arange(n_bins)
            labels = pd.qcut(labels, n_bins, labels=False, duplicates='drop')

        n_classes = classes.shape[0]
        n_total_val = int(frac_val * len(dataset))
        n_total_test = int(frac_test * len(dataset))

        class_counts = np.bincount(labels)
        class_idxs = np.split(np.argsort(labels, kind='mergesort'), np.cumsum(class_counts)[:-1])

        # Stratify samples from distribution for both validation and test sets. 
        # Remainder is the distribution of the training set. 
        # NOTE: The percentage of each class/bin should be equally represented in all 3 sets. 
        n_val_samples = _approximate_mode(class_counts, n_total_val)
        class_counts -= n_val_samples # remove validation samples from total count before sampling for test set
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
                               labels: Optional[np.ndarray]=None,
                               frac_train: Optional[float]=0.8, 
                               frac_val: Optional[float]=0.1, 
                               frac_test: Optional[float]=0.1, 
                               return_idxs: Optional[bool]=True, 
                               seed: Optional[int]=None, 
                               task_type: Optional[str]='auto',
                               n_bins: Optional[int]=10,
                               **kwargs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate indicies to split dataset into train, val, and test sets.

        Params:
        -------
        dataset: np.ndarray, size=(n,m)
            Dataset where `n` is the number of examples, and `m` is number of features.

        labels: np.ndarray or None, optional, default=None 
            Target labels of size=(n,). If `None`, function assumes that the last column of the 
            dataset is the labels. 

        frac_train: float, optional, default=0.8
            Fraction of data to be used in training. 

        frac_val: float, optional, default=0.1 
            Fraction of data to be used in validation.

        frac_test: float, optional, default=0.1
            Fraction of data to be used in testing.

        return_idxs: bool, optional, default=True
            If `True`, this function returns only indexes. If `False`, it returns splitted dataset.

        seed: int or None, optional, default=None
            Random state seed.

        task_type: bool, optional, default='auto'
            Type of training task. If `auto`, automatically determines task between classification 
            and regression based off labels data type. 

        n_bins: int, optional, default=10
            The number of bins to subset labels if task_type is continous.  

        Returns:
        --------
        splitted_data: Tuple[np.ndarray, np.ndarray, np.ndarray]
            The splitted data (train, val, and test) or their indicies.
        """
        return super(StratifiedSplitter, self).train_valid_test_split(dataset, frac_train, 
                                                                      frac_val, frac_test, 
                                                                      return_idxs, seed=seed, 
                                                                      labels=labels, n_bins=n_bins, 
                                                                      task_type=task_type, **kwargs)


    def train_valid_split(self, 
                          dataset: np.ndarray, 
                          labels: Optional[np.ndarray]=None,
                          frac_train: Optional[float]=0.8, 
                          frac_val: Optional[float]=0.2, 
                          return_idxs: Optional[bool]=True,  
                          seed: Optional[int]=None, 
                          task_type: Optional[str]='auto',
                          n_bins: Optional[int]=10,
                          **kwargs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate indicies to split dataset into train and val sets.

        Params:
        -------
        dataset: np.ndarray, size=(n,m)
            Dataset where `n` is the number of examples, and `m` is number of features.

        labels: np.ndarray or None, optional, default=None 
            Target labels of size=(n,). If `None`, function assumes that the last column of the 
            dataset is the labels. 

        frac_train: float, optional, default=0.8
            Fraction of data to be used in training. 

        frac_val: float, optional, default=0.1 
            Fraction of data to be used in validation.

        return_idxs: bool, optional, default=True
            If `True`, this function returns only indexes. If `False`, it returns splitted dataset.

        seed: int or None, optional, default=None
            Random state seed.

        task_type: bool, optional, default='auto'
            Type of training task. If `auto`, automatically determines task between classification 
            and regression based off labels data type. 

        n_bins: int, optional, default=10
            The number of bins to subset labels if task_type is continous.  

        Returns:
        --------
        splitted_data: Tuple[np.ndarray, np.ndarray]
            The splitted data (train and val) or their indicies.
        """
        return super(StratifiedSplitter, self).train_valid_split(dataset, frac_train, frac_val, 
                                                                 return_idxs, seed=seed, 
                                                                 n_bins=n_bins, labels=labels, 
                                                                 task_type=task_type, **kwargs)
