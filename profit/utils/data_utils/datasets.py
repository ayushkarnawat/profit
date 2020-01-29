import os
import json

from typing import List, Union

import h5py
import lmdb
import numpy as np
import pickle as pkl
import tensorflow as tf

import torch
from torch.utils.data import Dataset

from profit import backend as P


class TensorflowHDF5Dataset(tf.data.Dataset):
    """Parse (generic) HDF5 dataset into a `tf.data.Dataset` object, 
    which contains `tf.Tensor`s.  

    Params:
    -------
    path: str
        HDF5 file which contains dataset.
    """
    pass


class TensorflowLMDBDataset(object):
    """Parse (generic) HDF5 dataset into a `tf.data.Dataset` object, 
    which contains `tf.Tensor`s.
    
    Params:
    -------
    path: str
        LMDB file which contains dataset.
    """
    pass


class TensorflowNumpyDataset(object):
    """Parse (generic) numpy dataset into a `tf.data.Dataset` object, 
    which contains `tf.Tensor`s.
    
    Params:
    -------
    path: str
        Npz file which contains dataset.
    """
    pass


class TFRecordsDataset(object):
    """Parse (generic) tensorflow dataset into a `tf.data.Dataset` object, 
    which contains `tf.Tensor`s.
    
    Params:
    -------
    path: str
        TFRecords file which contains dataset.
    """
    pass


class TorchHDF5Dataset(Dataset):
    """Parse (generic) HDF5 dataset into a `torch.utils.data.Dataset` 
    object, which contains `torch.Tensor`s.

    Params:
    -------
    path: str
        HDF5 file which contains dataset.
    """

    def __init__(self, path: str):
        self.h5file = h5py.File(path, "r")
        self.keys = list(self.h5file.keys())

    def __len__(self) -> int:
        # # Check for same num of examples in the multiple ndarray's 
        # num_examples = [self.h5file.get(key).shape[0] if P.data_format() == "channels_first"
        #                 else self.h5file.get(key).shape[-1] for key in self.h5file]
        # assert num_examples[1:] == num_examples[:-1], \
        #     "Unequal num of examples - is the data format correct?"
        # return num_examples[0]
        return len(self.keys)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, List[torch.Tensor]]:
        # # Convert to pytorch tensors
        # if P.data_format() == "channels_first":
        #     sample = [torch.from_numpy(self.h5file.get(key)[idx]) 
        #               for key in self.h5file.keys()]
        # else: # channels_last
        #     sample = [torch.from_numpy(self.h5file.get(key)[...,idx]) 
        #               for key in self.h5file.keys()]
        # return sample[0] if len(sample) == 1 else sample

        # Convert to pytorch tensors
        sample_dict = json.loads(self.h5file.get(self.keys[idx])[()])
        sample = [torch.FloatTensor(arr) for arr in sample_dict.values()]
        return sample[0] if len(sample) == 1 else sample


class TorchLMDBDataset(Dataset):
    """Parse (generic) LMDB dataset into a `torch.utils.data.Dataset` 
    object, which contains `torch.Tensor`s.
    
    TODO: Handling of channels_first and channels_last! Need to modify 
    how db arrays and keys are saved within serializers.py to account 
    for both channels_first and channels_last! For now, each example 
    is only stored in first channel in the LMDB serializer.
    
    The above approach is wrong since it requires the user to properly 
    save the dataset in the format they want to read it in again, whether 
    that be channels_first or channels_last. Rather, we should decouple 
    the saving and loading (similar to numpy/hdf5) so that one doesn't 
    depend on the other.

    Params:
    -------
    path: str
        LMDB file which contains dataset.
    """

    def __init__(self, path: str):
        # Check whether directory or full filename is provided. If dir, check 
        # for "data.mdb" file within dir.
        isdir = os.path.isdir(path)
        if isdir:
            default_path = os.path.join(path, "data.mdb")
            assert os.path.isfile(default_path), "LMDB default file {} does " \
                "not exist!".format(default_path)
        else:
            assert os.path.isfile(path), "LMDB file {} does not exist!".format(path)

        self.db = lmdb.open(path, subdir=isdir, readonly=True, readahead=False)
        with self.db.begin() as txn, txn.cursor() as cursor:
            self.num_examples = txn.stat()['entries'] - 1
            self.keys = pkl.loads(cursor.get(b"__keys__"))

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int) -> Union[torch.Tensor, List[torch.Tensor]]:
        # Check for invalid indexing
        if idx > self.num_examples:
            raise IndexError(f"Index ({idx}) out of range (0-{self.num_examples})")
        
        with self.db.begin() as txn, txn.cursor() as cursor:
            ex = pkl.loads(cursor.get(self.keys[idx]))
            sample = [torch.from_numpy(arr) for arr in ex]
        return sample[0] if len(sample) == 1 else sample


class TorchNumpyDataset(Dataset):
    """Parse (generic) numpy dataset into a `torch.utils.data.Dataset` 
    object, which contains `torch.Tensor`s.

    Params:
    -------
    path: str
        Npz file which contains dataset.
    """

    def __init__(self, path: str):
        self.npzfile = np.load(path, allow_pickle=False)

    def __len__(self) -> int:
        # Check for same num of examples in the multiple ndarray's 
        num_examples = [arr.shape[0] if P.data_format() == "channels_first" 
                        else arr.shape[-1] for arr in self.npzfile.values()]
        assert num_examples[1:] == num_examples[:-1], \
            "Unequal num of examples - is the data format correct?"
        return num_examples[0]

    def __getitem__(self, idx: int) -> Union[torch.Tensor, List[torch.Tensor]]:
        # Convert to pytorch tensors
        if P.data_format() == "channels_first":
            sample = [torch.from_numpy(arr[idx]) for arr in self.npzfile.values()]
        else: # channels_last
            sample = [torch.from_numpy(arr[..., idx]) for arr in self.npzfile.values()]
        return sample[0] if len(sample) == 1 else sample


class TorchTFRecordsDataset(Dataset):
    """Parse (generic) tensorflow dataset into a `torch.utils.data.Dataset` 
    object, which contains `torch.Tensor`s.
    
    Ideally, this should be done without need for using tensorflow. See: 
    https://discuss.pytorch.org/t/read-dataset-from-tfrecord-format/16409

    Params:
    -------
    path: str
        TFRecords file which contains dataset.
    """

    def __init__(self, path: str):
        pass

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Union[torch.Tensor, List[torch.Tensor]]:
        raise NotImplementedError


if __name__ == "__main__":
    P.set_data_format('channels_first')
    torch_dataset = TorchHDF5Dataset("data/3gb1/processed/transformer_fitness/primary5.h5")
    # torch_dataset = TorchLMDBDataset("data/3gb1/processed/transformer_fitness/primary.mdb")
    # torch_dataset = TorchNumpyDataset("data/3gb1/processed/transformer_fitness/primary5.npz")
    # torch_dataset = TorchTFRecordsDataset("data/3gb1/processed/transformer_fitness/primary.tfrecords")
    # sample = torch_dataset[755]
    for i in range(len(torch_dataset)): 
        sample = torch_dataset[i] 
        print(i, [tensor.shape for tensor in sample])