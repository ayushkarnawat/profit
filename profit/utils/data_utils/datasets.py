import os
import pickle as pkl

from functools import partial
from typing import Dict, List, Tuple

import h5py
import lmdb
import numpy as np
import tensorflow as tf

import torch
from torch.utils.data import Dataset, IterableDataset

from profit.utils.data_utils.tfreader import tfrecord_loader


def TensorflowHDF5Dataset(path: str) -> tf.data.Dataset:
    """Parse (generic) HDF5 dataset into a `tf.data.Dataset` object,
    which contains `tf.Tensor`s.

    NOTE: The HDF5 file must contain an attribute named "saved_axis"
    which represents the axis where the `num_samples` are saved.

    Params:
    -------
    path: str
        HDF5 file which contains dataset.

    Returns:
    --------
    dataset: tf.data.Dataset
        Dataset loaded into its `tf.data.Dataset` form.
    """
    with h5py.File(path, "r") as h5file:
        # Move axis representing num_samples (aka saved_axis) to first axis
        saved_axis = h5file.attrs.get("saved_axis")
        data_dict = {key: np.moveaxis(arr[:], saved_axis, 0) for key, arr in h5file.items()}
    return tf.data.Dataset.from_tensor_slices(data_dict)


def TensorflowLMDBDataset(path: str) -> tf.data.Dataset:
    """Parse (generic) LMDB dataset into a `tf.data.Dataset` object,
    which contains `tf.Tensor`s.

    NOTE: The LMDB file must contain a key named "saved_axis" which
    represents the axis where the `num_samples` are saved.

    Params:
    -------
    path: str
        LMDB file which contains dataset.

    Returns:
    --------
    dataset: tf.data.Dataset
        Dataset loaded into its `tf.data.Dataset` form.
    """
    # Check whether directory or full filename is provided. If dir,
    # check for "data.mdb" file within dir.
    isdir = os.path.isdir(path)
    if isdir:
        default_path = os.path.join(path, "data.mdb")
        assert os.path.isfile(default_path), "LMDB default file {} " \
            "does not exist!".format(default_path)
    else:
        assert os.path.isfile(path), f"LMDB file {path} does not exist!"

    db = lmdb.open(path, subdir=isdir, readonly=True, readahead=False)
    with db.begin() as txn, txn.cursor() as cursor:
        # Move axis representing num_samples (aka saved_axis) to first axis
        saved_axis = pkl.loads(cursor.get(b"saved_axis"))
        data_dict = {key.decode(): np.moveaxis(pkl.loads(cursor.get(key)), saved_axis, 0)
                     for key in pkl.loads(cursor.get(b"__keys__"))}
    return tf.data.Dataset.from_tensor_slices(data_dict)


def TensorflowNumpyDataset(path: str) -> tf.data.Dataset:
    """Parse (generic) numpy dataset into a `tf.data.Dataset` object,
    which contains `tf.Tensor`s.

    NOTE: The npz file must contain a key named "saved_axis" which
    represents the axis where the `num_samples` are saved.

    Params:
    -------
    path: str
        Npz file which contains dataset.

    Returns:
    --------
    dataset: tf.data.Dataset
        Dataset loaded into its `tf.data.Dataset` form.
    """
    with np.load(path, allow_pickle=False) as npzfile:
        # Move axis representing num_samples (aka saved_axis) to first axis
        saved_axis = int(npzfile["saved_axis"])
        data_dict = {key: np.moveaxis(npzfile[key], saved_axis, 0)
                     for key in npzfile["__keys__"]}
    return tf.data.Dataset.from_tensor_slices(data_dict)


def TFRecordsDataset(path: str) -> tf.data.Dataset:
    """Parse (generic) TFRecords file into a `tf.data.Dataset` object,
    which contains `tf.Tensor`s.

    Params:
    -------
    path: str
        TFRecords file which contains dataset.

    Returns:
    --------
    dataset: tf.data.Dataset
        Dataset loaded into its `tf.data.Dataset` form.
    """
    def _deserialize(serialized: tf.Tensor, features: Dict[str, tf.io.FixedLenFeature],
                     **kwargs: Dict[str, List[int]]) -> Dict[str, Tuple[tf.Tensor, ...]]:
        """Deserialize an serialized example within the dataset.

        Params:
        -------
        serialized: tf.Tensor with dtype `string``
            Tensor containing a batch of binary serialized tf.Example protos.
        """
        parsed_example = tf.io.parse_single_example(serialized=serialized, \
            features=features)
        tensors = {}
        for name, tensor in parsed_example.items():
            if name.startswith("arr"):
                shape = kwargs.get("shape_{}".format(name.split("_")[-1]))
                tensors[name] = tf.reshape(tf.decode_raw(tensor, tf.float32), shape)
        return tensors

    # Determine what shapes each np.ndarray should be reshaped to.
    # Hack to allow saved flattened ndarray's to be reshaped properly.
    # NOTE: We could not properly convert each saved shape (which were
    # serialized as int64_lists and parsed as tf.Tensor) into lists so that
    # the ndarray's could be properly reshaped within _deserialize() above.
    serialized = next(tf.python_io.tf_record_iterator(path))
    example = tf.train.Example()
    example.ParseFromString(serialized)
    shapes = {name: list(example.features.feature[name].int64_list.value)
              for name in example.features.feature.keys() if name.startswith("shape")}

    # Define a dict with the data names and types we expect to find in
    # the TFRecords file. It is a quite cumbersome that this needs to be
    # specified again, because it could have been written in the header
    # of the TFRecords file instead.
    features = {}
    for name in example.features.feature.keys():
        if name.startswith("arr"):
            # NOTE: Assuming all arr_i's are saved as bytes_list
            features[name] = tf.io.FixedLenFeature([], tf.string)
        elif name.startswith("shape"):
            # NOTE: Assuming arr shape_i's are saved as int64_lists
            features[name] = tf.io.FixedLenFeature([], tf.int64)
        else:
            raise TypeError("Unknown dtype for {}.".format(name))

    # Parse serialized records into correctly shaped tensors/ndarray's
    dataset = tf.data.TFRecordDataset(path)
    return dataset.map(partial(_deserialize, features=features, **shapes))


class TorchHDF5Dataset(Dataset):
    """Parse (generic) HDF5 dataset into a `torch.utils.data.Dataset`
    object, which contains `torch.Tensor`s.

    NOTE: The HDF5 file must contain an attribute named "saved_axis"
    which represents the axis where the `num_samples` are saved.

    Params:
    -------
    path: str
        HDF5 file which contains dataset.
    """

    def __init__(self, path: str):
        with h5py.File(path, "r") as h5file:
            self.saved_axis = h5file.attrs.get("saved_axis")
            self.keys = list(h5file.keys())
            tensors = [torch.FloatTensor(arr[:]) for arr in h5file.values()]
        assert all(tensors[0].size(self.saved_axis) == tensor.size(self.saved_axis)
                   for tensor in tensors)
        self.tensors = dict(zip(self.keys, tensors))

    def __len__(self) -> int:
        return self.tensors[self.keys[0]].size(self.saved_axis)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {key: tensor[idx] if self.saved_axis == 0 else tensor[..., idx]
                for key, tensor in self.tensors.items()}


class TorchLMDBDataset(Dataset):
    """Parse (generic) LMDB dataset into a `torch.utils.data.Dataset`
    object, which contains `torch.Tensor`s.

    NOTE: The LMDB file must contain a key named "saved_axis" which
    represents the axis where the `num_samples` are saved.

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

        db = lmdb.open(path, subdir=isdir, readonly=True, readahead=False)
        with db.begin() as txn, txn.cursor() as cursor:
            self.saved_axis = pkl.loads(cursor.get(b"saved_axis"))
            self.keys = pkl.loads(cursor.get(b"__keys__"))
            tensors = [torch.from_numpy(pkl.loads(cursor.get(key)))
                       for key in self.keys]
        assert all(tensors[0].size(self.saved_axis) == tensor.size(self.saved_axis)
                   for tensor in tensors)
        self.tensors = dict(zip(self.keys, tensors))

    def __len__(self) -> int:
        return self.tensors[self.keys[0]].size(self.saved_axis)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {key.decode(): tensor[idx] if self.saved_axis == 0 else tensor[..., idx]
                for key, tensor in self.tensors.items()}


class TorchNumpyDataset(Dataset):
    """Parse (generic) numpy dataset into a `torch.utils.data.Dataset`
    object, which contains `torch.Tensor`s.

    NOTE: The npz file must contain a key named "saved_axis" which
    represents the axis where the `num_samples` are saved.

    Params:
    -------
    path: str
        Npz file which contains dataset.
    """
    def __init__(self, path: str):
        with np.load(path, allow_pickle=False) as npzfile:
            self.saved_axis = int(npzfile["saved_axis"])
            self.keys = list(npzfile["__keys__"])
            tensors = [torch.from_numpy(npzfile[key]) for key in self.keys]
        assert all(tensors[0].size(self.saved_axis) == tensor.size(self.saved_axis)
                   for tensor in tensors)
        self.tensors = dict(zip(self.keys, tensors))

    def __len__(self) -> int:
        return self.tensors[self.keys[0]].size(self.saved_axis)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {key: tensor[idx] if self.saved_axis == 0 else tensor[..., idx]
                for key, tensor in self.tensors.items()}


class TorchTFRecordsDataset(IterableDataset):
    """Parse (generic) tensorflow dataset into a `torch.utils.data.IterableDataset`
    object, which contains `torch.Tensor`s.

    NOTE: Uses example_pb2 to parse through the serialized records
    contained in the tfrecords file. This enables us to convert a
    tfrecords file to `torch.Tensor`s without the need to have the
    tf package installed. Useful down the line when we decouple the
    backends (aka not having one installed should not affect you ability
    to use the other backend).

    Params:
    -------
    path: str
        TFRecords file which contains dataset.
    """

    def __init__(self, path: str):
        self.tfrecord_path = path
        self.index_path = f"{path}_idx" if os.path.exists(f"{path}_idx") else None

    def __iter__(self) -> Dict[str, torch.Tensor]:
        # Shard/Chunk dataset if multiple workers are iterating over dataset
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            shard = worker_info.id, worker_info.num_workers
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
        else:
            shard = None

        # Reshape example into correctly shaped tensors/ndarray's
        records = tfrecord_loader(self.tfrecord_path, self.index_path, shard)
        for record in records:
            example = {}
            for key, value in record.items():
                if key.startswith("arr"):
                    shape = record.get("shape_{}".format(key.split("_")[-1]))
                    example[key] = np.reshape(value, newshape=shape)
            yield example
