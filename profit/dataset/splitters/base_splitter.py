from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np


class BaseSplitter(ABC):
    """Base class for performing splits."""

    @abstractmethod
    def _split(self,
               dataset: np.ndarray,
               frac_train: float = 0.75,
               frac_valid: float = 0.15,
               frac_test: float = 0.15,
               **kwargs: Dict[str, Any]):
        raise NotImplementedError


    def train_valid_test_split(self,
                               dataset: np.ndarray,
                               frac_train: float = 0.7,
                               frac_valid: float = 0.15,
                               frac_test: float = 0.15,
                               return_idxs: bool = True,
                               **kwargs: Dict[str, Any]) -> Tuple[np.ndarray, ...]:
        train_idxs, val_idxs, test_idxs = self._split(dataset, frac_train,
                                                      frac_valid, frac_test,
                                                      **kwargs)
        if return_idxs:
            return train_idxs, val_idxs, test_idxs
        return dataset[train_idxs], dataset[val_idxs], dataset[test_idxs]


    def train_valid_split(self,
                          dataset: np.ndarray,
                          frac_train: float = 0.85,
                          frac_valid: float = 0.15,
                          return_idxs: bool = True,
                          **kwargs: Dict[str, Any]) -> Tuple[np.ndarray, ...]:
        train_idxs, val_idxs, test_idxs = self._split(dataset, frac_train,
                                                      frac_valid, 0., **kwargs)

        assert len(test_idxs) == 0

        if return_idxs:
            return train_idxs, val_idxs
        return dataset[train_idxs], dataset[val_idxs]
