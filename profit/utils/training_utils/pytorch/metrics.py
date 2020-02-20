"""Performance metrics."""

from typing import List, Union

import numpy as np
import scipy as scp


def accuracy(y_true: Union[List[float], List[List[float]]],
             y_pred: Union[List[float], List[List[float]]]) -> float:
    if isinstance(y_true[0], float):
        # non-sequence case
        return np.mean(np.array(y_true) == np.array(y_pred))
    else:
        correct = 0
        total = 0
        for label, score in zip(y_true, y_pred):
            label_array = np.array(label)
            pred_array = np.array(score)
            mask = label_array != -1 # Only check positive labels
            is_correct = label_array[mask] == pred_array[mask]
            correct += is_correct.sum()
            total += is_correct.size
        return correct / total


def mae(y_true: Union[float, List[float], np.ndarray], 
        y_pred: Union[float, List[float], np.ndarray]) -> float:
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true: Union[float, List[float], np.ndarray], 
        y_pred: Union[float, List[float], np.ndarray]) -> float:
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    return np.mean(np.square(y_true - y_pred))


def rmse(y_true: Union[float, List[float], np.ndarray], 
         y_pred: Union[float, List[float], np.ndarray]) -> float:
    return np.sqrt(mse(y_true, y_pred))


def spearmanr(y_true: Union[float, List[float], np.ndarray], 
              y_pred: Union[float, List[float], np.ndarray]) -> float:
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    return scp.stats.spearmanr(y_true, y_pred).correlation