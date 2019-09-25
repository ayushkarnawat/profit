"""General utils."""

try:
    import pandas as pd
    use_pd = True
except ImportError:
    import csv
    use_pd = False

import numpy as np
from typing import Tuple


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
    if use_pd:
        df = pd.read_csv(filepath, sep=',')
        X = np.array(df[x_name].values, dtype=str)
        y = np.array(df[y_name].values, dtype=float)
    else:
        X,y = [], []
        with open(filepath) as file:
            reader = csv.DictReader(file)
            for row in reader:
                X.append(row[x_name])
                y.append(row[y_name])
        X = np.array(X, dtype=str)
        y = np.array(y, dtype=float)
    return X,y
