import numpy as np

from profit.dataset.splitter.base_splitter import BaseSplitter
from typing import Any, Dict, Optional, Tuple, Union


class RandomSplitter(BaseSplitter):
    """Class for random splitting of data."""


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
        seed = kwargs.get('seed', None)
        np.testing.assert_almost_equal(frac_train + frac_val + frac_test, 1.0)

        if seed is not None:
            # Keeping it in one line ensures that the same permutation is generated everytime
            shuffled = np.random.RandomState(seed).permutation(len(dataset))
        else:
            shuffled = np.random.permutation(len(dataset))
        train_size = int(len(dataset) * frac_train)
        val_size = int(len(dataset) * frac_val)
        return (shuffled[:train_size], shuffled[train_size:train_size + val_size],
                shuffled[train_size + val_size:])

    
    def train_valid_test_split(self, 
                               dataset: np.ndarray, 
                               frac_train: Optional[float]=0.8, 
                               frac_valid: Optional[float]=0.1, 
                               frac_test: Optional[float]=0.1,
                               return_idxs: Optional[bool]=True, 
                               seed: Union[int, None]=None, 
                               **kwargs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate indicies to split dataset into train, val, and test sets.

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

        return_idxs: bool, optional, default=True
            If `True`, this function returns only indexes. If `False`, it returns splitted dataset.

        seed: int or None, optional, default=None
            Random state seed.

        Returns:
        --------
        splitted_data: Tuple[np.ndarray, np.ndarray, np.ndarray]
            The splitted data (train, val, and test) or their indicies.
        """
        return super(RandomSplitter, self).train_valid_test_split(dataset, frac_train, frac_valid, 
                                                                  frac_test, return_idxs, seed=seed, 
                                                                  **kwargs)

    
    def train_valid_split(self, 
                          dataset: np.ndarray, 
                          frac_train: Optional[float]=0.8, 
                          frac_val: Optional[float]=0.2, 
                          return_idxs: Optional[bool]=True, 
                          seed: Union[int, None]=None, 
                          **kwargs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate indicies to split dataset into train and val sets.

        Params:
        -------
        dataset: np.ndarray, size=(n,m)
            Dataset where `n` is the number of examples, and `m` is number of features.

        frac_train: float, optional, default=0.8
            Fraction of data to be used in training. 

        frac_val: float, optional, default=0.1 
            Fraction of data to be used in validation.

        return_idxs: bool, optional, default=True
            If `True`, this function returns only indexes. If `False`, it returns splitted dataset.

        seed: int or None, optional, default=None
            Random state seed.

        Returns:
        --------
        splitted_data: Tuple[np.ndarray, np.ndarray]
            The splitted data (train and val) or their indicies.
        """
        return super(RandomSplitter, self).train_valid_split(dataset, frac_train, frac_val, 
                                                             return_idxs, seed=seed, **kwargs)


if __name__ == "__main__":
    X = np.random.random(size=(10,2))
    splitter  = RandomSplitter()
    X_train, X_val = splitter.train_valid_split(X, frac_train=0.8, frac_val=0.2, return_idxs=False, seed=None)
    print(X_train.shape, X_val.shape)
    