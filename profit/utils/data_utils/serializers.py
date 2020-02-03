import os
import json
import platform

from abc import abstractmethod, ABC
from functools import partial
from typing import Any, Dict, List, Tuple, Union

import h5py
import lmdb
import numpy as np
import pickle as pkl
import tensorflow as tf

from tqdm import tqdm
from torch.utils.data import Dataset as TorchDataset
from tensorflow.data import Dataset as TensorflowDataset

from profit import backend as P
from profit.utils.data_utils.datasets import TensorflowHDF5Dataset
from profit.utils.data_utils.datasets import TensorflowLMDBDataset
from profit.utils.data_utils.datasets import TensorflowNumpyDataset
from profit.utils.data_utils.datasets import TorchHDF5Dataset
from profit.utils.data_utils.datasets import TorchLMDBDataset
from profit.utils.data_utils.datasets import TorchNumpyDataset


class BaseSerializer(ABC):
    """Base serializer."""

    @staticmethod
    @abstractmethod
    def save(data: Any, path: str) -> None:
        """Save the data to file."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load(path: str) -> Any:
        """Load the dataset into its original shape/format."""
        raise NotImplementedError


class InMemorySerializer(BaseSerializer, ABC):
    """Serialize and load the dataset in memory."""

    @staticmethod
    @abstractmethod
    def save(data: Any, path: str) -> None:
        """Save the data to file."""
        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def load(path: str) -> Any:
        """Load the dataset (in memory) into its original shape/format."""
        raise NotImplementedError


class LazySerializer(BaseSerializer, ABC):
    """Serialize and load the dataset lazily."""

    @staticmethod
    @abstractmethod
    def save(data: Any, path: str) -> None:
        """Save the data to file."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load(path: str) -> Any:
        """Lazily load the dataset into its original shape/format."""
        raise NotImplementedError


class HDF5Serializer(InMemorySerializer):
    """Serialize ndarray's to HDF5 file.
    
    Note that HDF5 files are not that performant and do not (currently) 
    support lazy loading. It is better to use :class:`LMDBSerializer`.
    """

    @staticmethod
    def save(data: Union[np.ndarray, List[np.ndarray]], path: str) -> None:
        """Save data to .h5 (or .hdf5) file.
        
        Params:
        -------
        data: np.ndarray or list of np.ndarray
            The ndarray's to serialize.

        path: str
            Output HDF5 file.
        """
        if isinstance(data, np.ndarray):
            data = [data]

        with h5py.File(path, "w") as h5file:
            # Check for same num of examples in the multiple ndarray's
            shapes = [arr.shape for arr in data]
            axis = 0 if P.data_format() == "channels_first" else -1
            num_examples = [shape[axis] for shape in shapes]
            if num_examples[1:] != num_examples[:-1]:
                raise AssertionError(f"Unequal num of examples in {P.data_format()} " +
                    f"(axis={axis}): {shapes} - is the data format correct?")

            n_examples = num_examples[0]
            for idx in tqdm(range(n_examples), total=n_examples):
                # NOTE: Since numpy cannot serialize lists of np.arrays together 
                # for each individual example, we have to store them as a JSON 
                # object. Each example, which is denoted by a key, contains a 
                # dict with the key names being "arr_0", ..., "arr_n" based on 
                # which array it is referencing. Additionally, each ndarray has 
                # to be converted to lists of lists for proper storage.
                example = {f"arr_{i}": arr[idx].tolist() if P.data_format() == \
                    "channels_first" else arr[...,idx].tolist() for i,arr in enumerate(data)}
                key = u'{:08}'.format(idx).encode('ascii')

                # Convert dict to str so that json.loads() can recover 
                # info correctly when loading.
                h5file.create_dataset(name=key, data=json.dumps(example))


    @staticmethod
    def load(path: str, as_numpy: bool=False) -> Union[List[np.ndarray], \
            np.ndarray, TorchDataset, TensorflowDataset]:
        """Load the dataset.
        
        Params:
        -------
        path: str
            HDF5 file which contains dataset.

        as_numpy: bool, default=False
            If True, loads the dataset as a list of np.ndarray's (in 
            its original form). If False, loads the dataset in  
            P.backend()'s specified format.

        Returns:
        --------
        data: np.ndarray, torch.utils.data.Dataset, or tf.data.Dataset
            The dataset (either in its original shape/format or in 
            P.backend()'s specified format).
        """
        if as_numpy:
            dataset_dict = {}
            with h5py.File(path, "r") as h5file:
                for key in list(h5file.keys()):
                    example = json.loads(h5file.get(key)[()])
                    for name, arr in example.items():
                        arr = np.array(arr)
                        newshape = [1] + list(arr.shape) if P.data_format() == \
                            "channels_first" else list(arr.shape) + [1]
                        reshaped = np.reshape(arr, newshape=newshape)
                        # Concatenate individual examples together into one ndarray.
                        if name not in dataset_dict.keys():
                            dataset_dict[name] = reshaped
                        else:
                            dataset_dict[name] = np.vstack((dataset_dict.get(name), reshaped)) \
                                if P.data_format() == "channels_first" \
                                else np.hstack((dataset_dict.get(name), reshaped))
            # Extract np.ndarray's from dict and return in its original form
            data = [arr for arr in dataset_dict.values()]
            return data[0] if len(data) == 1 else data
        return TorchHDF5Dataset(path) if P.backend() == "pytorch" \
            else TensorflowHDF5Dataset(path)


class LMDBSerializer(LazySerializer):
    """Serialize ndarray's to a LMDB database.
    
    The keys are idxs, and the values are the list of serialized ndarrays.
    """

    @staticmethod
    def save(data: Union[np.ndarray, List[np.ndarray]], path: str, 
             write_frequency: int=1) -> None:
        """Save data to .lmdb/.mdb file.
        
        Params:
        -------
        data: np.ndarray or list of np.ndarray
            The ndarray's to serialize. The first channel of each 
            ndarray should contain the num of examples in the dataset.

        path: str
            Output LMDB directory or file.

        write_frequence: int, default=1
            The frequency to write back data to disk. Smaller value 
            reduces memory usage, at the cost of performance.
        """
        if isinstance(data, np.ndarray):
            data = [data]

        # Check for same num of examples in the multiple ndarray's
        shapes = [arr.shape for arr in data]
        axis = 0 if P.data_format() == "channels_first" else -1
        num_examples = [shape[axis] for shape in shapes]
        if num_examples[1:] != num_examples[:-1]:
            raise AssertionError(f"Unequal num of examples in {P.data_format()} " +
                f"(axis={axis}): {shapes} - is the data format correct?")
        
        # Check whether directory or full filename is provided. If dir, check 
        # for "data.mdb" file within dir.
        isdir = os.path.isdir(path)
        if isdir:
            assert not os.path.isfile(os.path.join(path, "data.mdb")), \
                "LMDB file {} exists!".format(os.path.join(path, "data.mdb"))
        else:
            assert not os.path.isfile(path), "LMDB file {} exists!".format(path)

        # It's OK to use super large map_size on Linux, but not on other platforms
        # See: https://github.com/NVIDIA/DIGITS/issues/206
        map_size = 1099511627776 * 2 if platform.system() == 'Linux' else 128 * 10**6
        db = lmdb.open(path, subdir=isdir, map_size=map_size, readonly=False, 
                       meminit=False, map_async=True) # need sync() at the end
        
        # Put data into lmdb, and doubling the size if full.
        # Ref: https://github.com/NVIDIA/DIGITS/pull/209/files
        def put_or_grow(txn, key, value):
            try:
                txn.put(key, value)
                return txn
            except lmdb.MapFullError:
                pass
            txn.abort()
            curr_size = db.info()['map_size']
            new_size = curr_size * 2
            print("Doubling LMDB map_size to {0:.2f}GB.".format(new_size / 10**9))
            db.set_mapsize(new_size)
            txn = db.begin(write=True)
            txn = put_or_grow(txn, key, value)
            return txn
        
        # NOTE: LMDB transaction is not exception-safe (even though it has a 
        # context manager interface).
        n_examples = num_examples[0]
        txn = db.begin(write=True)
        for idx in tqdm(range(n_examples), total=n_examples):
            example = [arr[idx] if P.data_format() == "channels_first" 
                       else arr[...,idx] for arr in data]
            txn = put_or_grow(txn, key=u'{:08}'.format(idx).encode('ascii'), 
                              value=pkl.dumps(example, protocol=-1))
            # NOTE: If we do not commit some examples before the db grows, 
            # those samples do not get saved. As such, for robustness, we 
            # choose write_frequency=1 (at the cost of performance).
            if (idx + 1) % write_frequency == 0:
                txn.commit()
                txn = db.begin(write=True)
        txn.commit() # commit all remaining serialized examples
        
        # Add all keys used (in this case it is just the idxs)
        keys = [u'{:08}'.format(k).encode('ascii') for k in range(n_examples)]
        with db.begin(write=True) as txn:
            txn = put_or_grow(txn, key=b'__keys__', value=pkl.dumps(keys, protocol=-1))

        print("Flushing database ...")
        db.sync()
        db.close()


    @staticmethod
    def load(path: str, as_numpy: bool=False) -> Union[List[np.ndarray], \
            np.ndarray, TorchDataset, TensorflowDataset]:
        """Load the dataset.
        
        Params:
        -------
        path: str
            LMDB file which contains dataset.

        as_numpy: bool, default=False
            If True, loads the dataset as a list of np.ndarray's (in 
            its original form). If False, loads the dataset in  
            P.backend()'s specified format.

        Returns:
        --------
        data: np.ndarray, torch.utils.data.Dataset, or tf.data.Dataset
            The dataset (either in its original shape/format or in 
            P.backend()'s specified format).
        """
        # Check whether directory or full filename is provided. If dir, check 
        # for "data.mdb" file within dir.
        isdir = os.path.isdir(path)
        if isdir:
            default_path = os.path.join(path, "data.mdb")
            assert os.path.isfile(default_path), "LMDB default file {} does " \
                "not exist!".format(default_path)
        else:
            assert os.path.isfile(path), "LMDB file {} does not exist!".format(path)
        
        if as_numpy:
            db = lmdb.open(path, subdir=isdir, readonly=True)
            dataset_dict = {}
            with db.begin() as txn, txn.cursor() as cursor:
                keys = pkl.loads(cursor.get(b"__keys__"))
                for key in keys:
                    example = pkl.loads(cursor.get(key))
                    for idx, arr in enumerate(example):
                        name = "arr_{}".format(idx)
                        newshape = [1] + list(arr.shape) if P.data_format() == \
                            "channels_first" else list(arr.shape) + [1]
                        reshaped = np.reshape(arr, newshape=newshape)
                        # Concatenate individual examples together into one ndarray.
                        if name not in dataset_dict.keys():
                            dataset_dict[name] = reshaped
                        else:
                            dataset_dict[name] = np.vstack((dataset_dict.get(name), reshaped)) \
                                if P.data_format() == "channels_first" \
                                else np.hstack((dataset_dict.get(name), reshaped))
            # Extract np.ndarray's from dict and return in its original form
            data = [arr for arr in dataset_dict.values()]
            return data[0] if len(data) == 1 else data
        return TorchLMDBDataset(path) if P.backend() == "pytorch" \
            else TensorflowLMDBDataset(path)


class NumpySerializer(InMemorySerializer):
    """Serialize ndarray's to a npz dict.
    
    Note that npz files do not support lazy loading and are >10x slower 
    than LMDB/TFRecord serializers. Use :class:`LMDBSerializer` instead.
    """

    @staticmethod
    def save(data: Union[np.ndarray, List[np.ndarray]], path: str, 
             compress: bool=True) -> None:
        """Save data to .npz file.
        
        Params:
        -------
        data: np.ndarray or list of np.ndarray
            The ndarray's to serialize.

        path: str
            Output npz file.

        compress: bool, default=True
            If True, uses gzip to compress the file. If False, no 
            compression is performed.
        """
        if isinstance(data, np.ndarray):
            data = [data]

        # Check for same num of examples in the multiple ndarray's
        shapes = [arr.shape for arr in data]
        axis = 0 if P.data_format() == "channels_first" else -1
        num_examples = [shape[axis] for shape in shapes]
        if num_examples[1:] != num_examples[:-1]:
            raise AssertionError(f"Unequal num of examples in {P.data_format()} " +
                f"(axis={axis}): {shapes} - is the data format correct?")

        dataset_dict = {}
        n_examples = num_examples[0]
        for idx in tqdm(range(n_examples), total=n_examples):
            # NOTE: Since numpy cannot serialize lists of np.arrays together 
            # for each individual example, we have to store them as a dict 
            # object. Each example, which is denoted by a key, contains a 
            # dict with the key names being "arr_0", ..., "arr_n" based on 
            # which array it is referencing. Additionally, each ndarray has 
            # to be converted to lists of lists for proper storage.
            example = {f"arr_{i}": arr[idx].tolist() if P.data_format() == \
                "channels_first" else arr[...,idx].tolist() for i,arr in enumerate(data)}
            key = u'{:08}'.format(idx)
            dataset_dict[key] = pkl.dumps(example)

        # Save each example (denoted by key) seperately
        np.savez_compressed(path, **dataset_dict) if compress \
            else np.savez(path, **dataset_dict)
        

    @staticmethod
    def load(path: str, as_numpy: bool=False) -> Union[List[np.ndarray], \
            np.ndarray, TorchDataset, TensorflowDataset]:
        """Load the dataset.
        
        Params:
        -------
        path: str
            Npz file which contains dataset.

        as_numpy: bool, default=False
            If True, loads the dataset as a list of np.ndarray's (in 
            its original form). If False, loads the dataset in  
            P.backend()'s specified format.

        Returns:
        --------
        data: np.ndarray, torch.utils.data.Dataset, or tf.data.Dataset
            The dataset (either in its original shape/format or in 
            P.backend()'s specified format).
        """
        if as_numpy:
            dataset_dict = {}
            with np.load(path, allow_pickle=False) as npzfile:
                for key in list(npzfile.keys()):
                    example = pkl.loads(npzfile.get(key))
                    for name, arr in example.items():
                        arr = np.array(arr)
                        newshape = [1] + list(arr.shape) if P.data_format() == \
                            "channels_first" else list(arr.shape) + [1]
                        reshaped = np.reshape(arr, newshape=newshape)
                        # Concatenate individual examples together into one ndarray.
                        if name not in dataset_dict.keys():
                            dataset_dict[name] = reshaped
                        else:
                            dataset_dict[name] = np.vstack((dataset_dict.get(name), reshaped)) \
                                if P.data_format() == "channels_first" \
                                else np.hstack((dataset_dict.get(name), reshaped))
            # Extract np.ndarray's from dict and return in its original form
            data = [arr for arr in dataset_dict.values()]
            return data[0] if len(data) == 1 else data
        return TorchNumpyDataset(path) if P.backend() == "pytorch" \
            else TensorflowNumpyDataset(path)


class TFRecordsSerializer(LazySerializer):
    """Serialize np.ndarray's to bytes and write to TFRecords file.
    
    Note that TFRecords does not support random access and is in fact 
    not very performant. It's better to use :class:`LMDBSerializer`.
    """

    @staticmethod
    def save(data: Union[np.ndarray, List[np.ndarray]], path: str) -> None:
        """Save data to .tfrecords file.
        
        Saves each np.ndarray under default names `arr_0`, ..., `arr_n` 
        and its associated shapes as `shape_0`, ..., `shape_n`.

        NOTE: TFRecords flatten each ndarray before saving them as a 
        bytes_list feature (thus losing array's shape metadata). To 
        combat this, we save each ndarray's shape dims (as int64_list 
        feature) and reshape them accordingly when loading.
        
        Params:
        -------
        data: np.ndarray or list of np.ndarray
            The ndarray's to serialize. The first channel of each 
            ndarray should contain the num of examples in the dataset.

        path: str
            Output TFRecords file.
        """
        def _bytes_feature(value: Union[str, bytes]):
            """Returns a bytes_list from a string / byte."""
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy() # BytesList won't unpack string from an EagerTensor.
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _float_feature(value: float):
            """Returns a float_list from a float / double."""
            if not isinstance(value, (list, np.ndarray)):
                value = [value] # FloatList won't unpack unless it is an list/np.array.
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))

        def _int64_feature(value: Union[bool, int]):
            """Returns an int64_list from a bool / enum / int / uint."""
            if not isinstance(value, (list, np.ndarray)):
                value = [value] # Int64List won't unpack, unless it is an list/np.array.
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

        def _serialize(example: Dict[str, Dict[str, Any]]) -> str:
            """Serialize an example within the dataset."""
            dset_item = {}
            for feature in example.keys():
                dset_item[feature] = example[feature]["_type"](example[feature]["data"])
                features = tf.train.Features(feature=dset_item)
                example_proto = tf.train.Example(features=features)
            return example_proto.SerializeToString()

        if isinstance(data, np.ndarray):
            data = [data]

        # Check for same num of examples in the multiple ndarray's
        # TODO: Make tensorflow support different data formats, currently only 
        # supports channels_first. NOTE: Will break if using channels_last!
        shapes = [arr.shape for arr in data]
        axis = 0 if P.data_format() == "channels_first" else -1
        num_examples = [shape[axis] for shape in shapes]
        if num_examples[1:] != num_examples[:-1]:
            raise AssertionError(f"Unequal num of examples in {P.data_format()} " +
                f"(axis={axis}): {shapes} - is the data format correct?")

        # Add shapes of each array in the dataset (for a single example). Hack 
        # to allow serialized data to be reshaped properly when loaded.
        dataset = {"arr_{}".format(idx): arr for idx, arr in enumerate(data)}
        dataset.update({"shape_{}".format(idx): np.array(arr.shape[1:]) 
                        for idx, arr in enumerate(data)})

        # Write serialized example(s) into the dataset
        n_examples = data[0].shape[0]
        with tf.io.TFRecordWriter(path) as writer:
            for row in tqdm(range(n_examples), total=n_examples):
                # NOTE: tobytes() flattens an ndarray. We have to flatten it 
                # because tf _bytes_feature() only takes in bytes. To combat 
                # this, we save each ndarray's shape as well (see above).
                example = {}
                for key, nparr in dataset.items():
                    # Save metadata about the array (aka shape) as int64 feature
                    if key.startswith("shape"):
                        example[key] = {"data": nparr, "_type": _int64_feature}
                    else:
                        example[key] = {"data": nparr[row].tobytes(), 
                                        "_type": _bytes_feature}
                writer.write(_serialize(example))

        # TODO: NEW ONE BELOW. Accounts for channels_first and channels_last!

        # Add shapes of each array in the dataset (for a single example). Hack 
        # to allow serialized data to be reshaped properly when loaded.
        shapes = {f"shape_{idx}": np.array(arr.shape[1:]) if P.data_format() == "channels_first" 
                  else np.array(arr.shape[:-1]) for idx, arr in enumerate(data)}
        dataset = {"arr_{}".format(idx): arr for idx, arr in enumerate(data)}
        dataset.update(shapes)

        # Write serialized example(s) into the dataset
        n_examples = data[0].shape[axis]
        with tf.io.TFRecordWriter(path) as writer:
            for row in tqdm(range(n_examples), total=n_examples):
                # NOTE: tobytes() flattens an ndarray. We have to flatten it 
                # because tf _bytes_feature() only takes in bytes. To combat 
                # this, we save each ndarray's shape as well (see above).
                example = {}
                for key, nparr in dataset.items():
                    # Save metadata about the array (aka shape) as int64 feature
                    if key.startswith("shape"):
                        example[key] = {"data": nparr, "_type": _int64_feature}
                    else:
                        example[key] = {"data": nparr[row].tobytes() if P.data_format() \
                            == "channels_first" else nparr[...,row].tobytes(), 
                                        "_type": _bytes_feature}
                writer.write(_serialize(example))


    @staticmethod
    def load(path: str, as_numpy: bool=False) -> Union[np.ndarray, \
            List[np.ndarray], tf.data.TFRecordDataset]:
        """Load the dataset.

        Assumes that each np.ndarray is saved under default names 
        `arr_0`, ..., `arr_n` and its associated shapes as `shape_0`, 
        ..., `shape_n`.
        
        Params:
        -------
        path: str
            TFRecords file which contains the dataset.

        as_numpy: bool, default=False
            If True, loads the dataset as a list of np.ndarray's (in 
            its original form). If False, loads it as a TFRecordDataset.

        Returns:
        --------
        data: np.ndarray or list of np.ndarray or tf.data.TFRecordDataset
            The dataset (either in its original shape/format or as a 
            TFRecordDataset).
        """
        def _deserialize(serialized: tf.Tensor, features: Dict[str, tf.io.FixedLenFeature], 
                         **kwargs: Dict[str, List[int]]) -> Tuple[tf.Tensor, ...]:
            """Deserialize an serialized example within the dataset.
            
            Params:
            -------
            serialized: tf.Tensor with dtype `string``
                Tensor containing a batch of binary serialized tf.Example protos.
            """
            parsed_example = tf.io.parse_single_example(serialized=serialized, \
                features=features)
            tensors = []
            for name, tensor in parsed_example.items():
                if name.startswith("arr"):
                    shape = kwargs.get("shape_{}".format(name.split("_")[-1]))
                    arr_i = tf.reshape(tf.decode_raw(tensor, tf.float32), shape)
                    tensors.append(arr_i)
            return tuple(tensors)
            
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
        if not as_numpy:
            dataset = tf.data.TFRecordDataset(path)
            data = dataset.map(partial(_deserialize, features=features, **shapes))
        else:
            dataset_dict = {}
            for serialized in tf.python_io.tf_record_iterator(path):
                example = tf.train.Example()
                example.ParseFromString(serialized)
                for name, tf_feature in example.features.feature.items():
                    if name.startswith("arr"):
                        parsed_data = np.frombuffer(tf_feature.bytes_list.value[0], np.float)
                        newshape = shapes.get("shape_{}".format(name.split("_")[-1]))
                        reshaped = np.reshape(parsed_data, newshape=[1] + newshape)
                        # Concatenate individual examples together into one ndarray.
                        # NOTE: Value of shape[0] (aka first channel) denotes the total 
                        # num of examples in the dataset.
                        if name not in dataset_dict.keys():
                            dataset_dict[name] = reshaped
                        else:
                            dataset_dict[name] = np.vstack((dataset_dict.get(name), reshaped))
            # Extract np.ndarray's from dict and return in its original form
            data = [arr for arr in dataset_dict.values()]
            if len(data) == 1:
                data = data[0]
        return data