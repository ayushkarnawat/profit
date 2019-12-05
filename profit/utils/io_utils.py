"""I/O utils."""

try:
    import pandas as pd
except ImportError:
    import csv

import os
import numpy as np
from typing import Tuple


class DownloadError(Exception):
    pass


def load_csv(filepath: str, 
             x_name: str, 
             y_name: str) -> Tuple[int, int]:
    """
    Loads the raw dataset.

    Params:
    -------
    filepath: str
        Path where the dataset is located.

    x_name: str
        Column name of the actual data X.

    y_name: str
        Column name of the labels in the dataset.

    Returns:
    --------
    X: np.ndarray
        Extracted data.

    y: np.ndarray
        Extracted labels corresponding to the dataset X. 
    """
    # If pandas exists, then use it, otherwise use csv
    try:
        df = pd.read_csv(filepath, sep=',')
        X = np.array(df[x_name].values, dtype=str)
        y = np.array(df[y_name].values, dtype=float)
    except NameError:
        X,y = [], []
        with open(filepath) as file:
            reader = csv.DictReader(file)
            for row in reader:
                X.append(row[x_name])
                y.append(row[y_name])
        X = np.array(X, dtype=str)
        y = np.array(y, dtype=float)
    return X,y


def maybe_create_dir(path: str) -> str:
    """Create directory if it doesn't already exist. 
    
    Params:
    -------
    path: str
        Full path location to save data.

    Returns:
    --------
    save_path: str
        Full path to where directory was created.
    """
    # If there is filename extension, remove filename with extension
    save_path = os.path.expanduser(path)
    save_dir = os.path.split(save_path)[0] if os.path.splitext(save_path)[1] else save_path
    if not os.path.isdir(save_dir):
        print('Creating directory `{0:s}`'.format(save_dir))
        os.makedirs(save_dir)
    return save_path
    