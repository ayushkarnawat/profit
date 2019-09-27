import pytest
import numpy as np

from profit.dataset.splitters.stratified_splitter import StratifiedSplitter


@pytest.fixture
def cls_dataset():
    # Create dataset with 10 features and last column as labels
    X = np.random.random((30, 10))
    y = np.concatenate([np.zeros(20), np.ones(10)]).astype(np.int)
    return np.concatenate([X, y.reshape(-1,1)], axis=1)


@pytest.fixture
def cls_label():
    y = np.concatenate([np.zeros(20), np.ones(10)]).astype(np.int)
    return y


@pytest.fixture
def reg_dataset():
    X = np.random.random((100, 10))
    y = np.random.random(size=100).astype(np.float) * 10 # range[0,10]
    return np.concatenate([X, y.reshape(-1,1)], axis=1)


def test_classification_split(cls_dataset):
    splitter = StratifiedSplitter()

    # Split using default values: 0.8 for train, 0.1 for val, and 0.1 for test
    train, valid, test = splitter.train_valid_test_split(cls_dataset, return_idxs=False)
    assert type(train) == np.ndarray
    assert train.shape[0] == 24
    assert valid.shape[0] == 3
    assert test.shape[0] == 3

    # Each set should contain the same ratio of positive to negative labels
    # For our ex, this is 1/3 true labels and 2/3 false labels, same ratio as full dataset
    assert (train[:,-1] == 0).sum() == 16 
    assert (train[:,-1] == 1).sum() == 8

    assert (valid[:,-1] == 0).sum() == 2
    assert (valid[:,-1] == 1).sum() == 1
    
    assert (test[:,-1] == 0).sum() == 2
    assert (test[:,-1] == 1).sum() == 1

    # Split using 0.5 for train, 0.3 for val, and 0.2 for test
    train, valid, test = splitter.train_valid_test_split(cls_dataset, frac_train=0.5, frac_val=0.3, 
                                                         frac_test=0.2, return_idxs=False)
    assert type(train) == np.ndarray
    assert train.shape[0] == 15
    assert valid.shape[0] == 9
    assert test.shape[0] == 6

    # Each set should contain the same ratio of positive to negative labels
    # For our ex, this is 1/3 true labels and 2/3 false labels, same ratio as full dataset
    assert (train[:,-1] == 0).sum() == 10
    assert (train[:,-1] == 1).sum() == 5

    assert (valid[:,-1] == 0).sum() == 6
    assert (valid[:,-1] == 1).sum() == 3
    
    assert (test[:,-1] == 0).sum() == 4
    assert (test[:,-1] == 1).sum() == 2


def test_regression_split(reg_dataset):
    splitter = StratifiedSplitter()

    # Split using default values: 0.8 for train, 0.1 for val, and 0.1 for test
    train, valid, test = splitter.train_valid_test_split(reg_dataset, return_idxs=False)
    assert type(train) == np.ndarray
    assert train.shape[0] == 80
    assert valid.shape[0] == 10
    assert test.shape[0] == 10
    assert 4.5 < train[:, -1].mean() < 5.5
    assert 4.5 < valid[:, -1].mean() < 5.5
    assert 4.5 < test[:, -1].mean() < 5.5

    # Split using 0.5 for train, 0.3 for val, and 0.2 for test
    train, valid, test = splitter.train_valid_test_split(reg_dataset, frac_train=0.5, frac_val=0.3, 
                                                         frac_test=0.2, return_idxs=False)
    assert type(train) == np.ndarray
    assert train.shape[0] == 50
    assert valid.shape[0] == 30
    assert test.shape[0] == 20
    assert 4.5 < train[:, -1].mean() < 5.5
    assert 4.5 < valid[:, -1].mean() < 5.5
    assert 4.5 < test[:, -1].mean() < 5.5
    