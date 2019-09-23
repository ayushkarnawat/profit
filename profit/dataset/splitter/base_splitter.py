import numpy as np
from typing import Any, Dict, Optional, Tuple


class BaseSplitter(object):
    """Base class for doing splits."""

    def _split(self, 
               dataset: np.ndarray, 
               frac_train: Optional[float]=0.75, 
               frac_val: Optional[float]=0.15, 
               frac_test: Optional[float]=0.15, 
               **kwargs: Dict[str, Any]):
        raise NotImplementedError

    
    def train_valid_test_split(self, 
                               dataset: np.ndarray, 
                               frac_train: Optional[float]=0.7, 
                               frac_valid: Optional[float]=0.15, 
                               frac_test: Optional[float]=0.15,
                               return_idxs: Optional[bool]=True,  
                               **kwargs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        train_idxs, val_idxs, test_idxs = self._split(dataset, frac_train, 
                                                      frac_valid, frac_test, 
                                                      **kwargs)
        if return_idxs:
            return train_idxs, val_idxs, test_idxs
        else:
            return dataset[train_idxs], dataset[val_idxs], dataset[test_idxs]


    def train_valid_split(self, 
                          dataset: np.ndarray, 
                          frac_train: Optional[float]=0.85, 
                          frac_valid: Optional[float]=0.15, 
                          return_idxs: Optional[bool]=True, 
                          **kwargs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        
        train_idxs, val_idxs, test_idxs = self._split(dataset, frac_train, 
                                                      frac_valid, 0., **kwargs)
        
        assert len(test_idxs) == 0

        if return_idxs:
            return train_idxs, val_idxs
        else:
            return dataset[train_idxs], dataset[val_idxs]
