import pytest
import numpy as np

from profit.dataset.splitters.random_splitter import RandomSplitter


@pytest.fixture
def dataset():
    # Create dataset with 10 features and last column as labels
    X = np.random.random((30, 10))
    y = np.concatenate([np.zeros(20), np.ones(10)]).astype(np.int)
    return np.concatenate([X, y.reshape(-1,1)], axis=1)


def test_train_valid_test_split(dataset):
    splitter = RandomSplitter()
    train_idx, valid_idx, test_idx = splitter.train_valid_test_split(dataset)
    assert type(train_idx) == np.ndarray
    assert train_idx.shape[0] == 24
    assert valid_idx.shape[0] == 3
    assert test_idx.shape[0] == 3

    train_idx, valid_idx, test_idx = splitter.train_valid_test_split(dataset, 0.5, 0.3, 0.2)
    assert type(train_idx) == np.ndarray
    assert train_idx.shape[0] == 15
    assert valid_idx.shape[0] == 9
    assert test_idx.shape[0] == 6


def test_train_valid_split(dataset):
    splitter = RandomSplitter()
    train_idx, valid_idx = splitter.train_valid_split(dataset)
    assert type(train_idx) == np.ndarray
    assert train_idx.shape[0] == 24
    assert valid_idx.shape[0] == 6

    train_idx, valid_idx = splitter.train_valid_split(dataset, 0.6, 0.4)
    assert type(train_idx) == np.ndarray
    assert train_idx.shape[0] == 18
    assert valid_idx.shape[0] == 12
    