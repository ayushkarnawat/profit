"""Create serialized representations of np.ndarrays into specified
filetypes (HDF5, LMDB, NP, TFRecords).

For HDF5, LMDB, and NP files, each ndarray gets saved in its own key
under default names `arr_0`, ..., `arr_n`. For TFRecord files, each
record (aka row/col in the ndarrays) is serialized into its own key.

This is done for performance reasons. When loading the dataset (either
into `torch.utils.data.Dataset` or `tf.data.Dataset` objects), it is
significantly faster to query each array and slice them as tensors
rather than load each individual example and concatenate them.

TODO: Perhaps, if the number of ndarrays to be saved is large, then the
opposite might be true. However, since this requires further testing,
let's leave that discussion for another day.

TODO: Should TFRecords files also just save each ndarray into its own
key rather than each example begin serialized individually? This might
not work when attempting to append new examples into the file, since we
will have to ensure that each record has the same shape as the previous
example(s).
"""

import os
import struct
import platform
import pickle as pkl

from abc import abstractmethod, ABC
from typing import Any, Dict, List, Union

import h5py
import lmdb
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from tensorflow.data import Dataset as TFDataset
from torch.utils.data import Dataset as TorchDataset

from profit import backend as P
from profit.utils.data_utils.datasets import TensorflowHDF5Dataset
from profit.utils.data_utils.datasets import TensorflowLMDBDataset
from profit.utils.data_utils.datasets import TensorflowNumpyDataset
from profit.utils.data_utils.datasets import TFRecordsDataset
from profit.utils.data_utils.datasets import TorchHDF5Dataset
from profit.utils.data_utils.datasets import TorchLMDBDataset
from profit.utils.data_utils.datasets import TorchNumpyDataset
from profit.utils.data_utils.datasets import TorchTFRecordsDataset


class BaseSerializer(ABC):
    """Base serializer."""

    @staticmethod
    @abstractmethod
    def save(data: Any, path: str, **kwargs: Dict[str, Any]) -> None:
        """Save the data to file."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load(path: str, as_numpy: bool = False) -> Any:
        """Load the dataset into its original shape/format."""
        raise NotImplementedError


class InMemorySerializer(BaseSerializer, ABC):
    """Serialize and load the dataset in memory."""

    @staticmethod
    @abstractmethod
    def save(data: Any, path: str, **kwargs: Dict[str, Any]) -> None:
        """Save the data to file."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load(path: str, as_numpy: bool = False) -> Any:
        """Load the dataset (in memory) into its original shape/format."""
        raise NotImplementedError


class LazySerializer(BaseSerializer, ABC):
    """Serialize and load the dataset lazily."""

    @staticmethod
    @abstractmethod
    def save(data: Any, path: str, **kwargs: Dict[str, Any]) -> None:
        """Save the data to file."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load(path: str, as_numpy: bool = False) -> Any:
        """Lazily load the dataset into its original shape/format."""
        raise NotImplementedError


class HDF5Serializer(InMemorySerializer):
    """Serialize ndarray's to HDF5 file.

    Note that HDF5 files are not that performant and do not (currently)
    support lazy loading. It is better to use :class:`LMDBSerializer`.
    """

    @staticmethod
    def save(data: Union[np.ndarray, List[np.ndarray]], path: str,
             compress: bool = True) -> None:
        """Save data to .h5 (or .hdf5) file.

        Params:
        -------
        data: np.ndarray or list of np.ndarray
            The ndarray's to serialize.

        path: str
            Output HDF5 file.

        compress: bool, default=True
            If True, uses gzip to compress the file. If False, no
            compression is performed.
        """
        if isinstance(data, np.ndarray):
            data = [data]

        # Convert np.ndarray's dtype to their 32 counterparts
        data = [arr.astype(arr.dtype.kind) for arr in data]

        # Check for same num of examples in the multiple ndarray's
        axis = 0 if P.data_format() == "batch_first" else -1
        assert all(data[0].shape[axis] == arr.shape[axis] for arr in data), \
            (f"Unequal num of examples in {P.data_format()} (axis={axis}): "
             f"{[arr.shape for arr in data]} - is the data format correct?")

        with h5py.File(path, "w") as h5file:
            # Save ndarrays and axis where num_samples is represented
            h5file.attrs.create("saved_axis", axis)
            for idx, arr in enumerate(data):
                h5file.create_dataset(name=f"arr_{idx}", data=arr, chunks=True,
                                      compression="gzip" if compress else None)


    @staticmethod
    def load(path: str, as_numpy: bool = False) -> Union[np.ndarray, \
            List[np.ndarray], TFDataset, TorchDataset]:
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
            with h5py.File(path, "r") as h5file:
                saved_axis = h5file.attrs.get("saved_axis")
                out_axis = 0 if P.data_format() == "batch_first" else -1
                data = [np.moveaxis(arr[:], saved_axis, out_axis)
                        for arr in h5file.values()]
            return data[0] if len(data) == 1 else data
        return TorchHDF5Dataset(path) if P.backend() == "pytorch" \
            else TensorflowHDF5Dataset(path)


class LMDBSerializer(LazySerializer):
    """Serialize ndarray's to a LMDB database.

    The keys are idxs, and the values are the list of serialized ndarrays.
    """

    @staticmethod
    def save(data: Union[np.ndarray, List[np.ndarray]], path: str,
             write_frequency: int = 1) -> None:
        """Save data to .lmdb/.mdb file.

        Params:
        -------
        data: np.ndarray or list of np.ndarray
            The ndarray's to serialize.

        path: str
            Output LMDB directory or file.

        write_frequence: int, default=1
            The frequency to write back data to disk. Smaller value
            reduces memory usage, at the cost of performance.
        """
        if isinstance(data, np.ndarray):
            data = [data]

        # Convert np.ndarray's dtype to their 32 counterparts
        data = [arr.astype(arr.dtype.kind) for arr in data]

        # Check for same num of examples in the multiple ndarray's
        axis = 0 if P.data_format() == "batch_first" else -1
        assert all(data[0].shape[axis] == arr.shape[axis] for arr in data), \
            (f"Unequal num of examples in {P.data_format()} (axis={axis}): "
             f"{[arr.shape for arr in data]} - is the data format correct?")

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
            print(f"Doubling LMDB map_size to {new_size / 10**9:.2f} GB.")
            db.set_mapsize(new_size)
            txn = db.begin(write=True)
            txn = put_or_grow(txn, key, value)
            return txn

        # NOTE: LMDB transaction is not exception-safe (even though it has a
        # context manager interface).
        txn = db.begin(write=True)
        for idx, arr in enumerate(data):
            key = f"arr_{idx}".encode('ascii')
            txn = put_or_grow(txn, key=key, value=pkl.dumps(arr, protocol=-1))
            # NOTE: If we do not commit some ndarrays before the db grows,
            # those do not get saved. As such, for robustness, we choose
            # write_frequency=1 (at the cost of performance).
            if (idx + 1) % write_frequency == 0:
                txn.commit()
                txn = db.begin(write=True)
        txn.commit() # commit all remaining serialized ndarrays

        # Add all keys used (in this case it is just the array names) and axis
        # where num_samples is represented
        keys = [f'arr_{idx}'.encode('ascii') for idx in range(len(data))]
        with db.begin(write=True) as txn:
            txn = put_or_grow(txn, key=b'__keys__', value=pkl.dumps(keys, protocol=-1))
            txn = put_or_grow(txn, key=b'saved_axis', value=pkl.dumps(axis, protocol=-1))

        print("Flushing database ...")
        db.sync()
        db.close()


    @staticmethod
    def load(path: str, as_numpy: bool = False) -> Union[np.ndarray, \
            List[np.ndarray], TFDataset, TorchDataset]:
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
            out_axis = 0 if P.data_format() == "batch_first" else -1
            db = lmdb.open(path, subdir=isdir, readonly=True)
            with db.begin() as txn, txn.cursor() as cursor:
                saved_axis = pkl.loads(cursor.get(b"saved_axis"))
                data = [np.moveaxis(pkl.loads(cursor.get(key)), saved_axis, out_axis)
                        for key in pkl.loads(cursor.get(b"__keys__"))]
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
             compress: bool = True) -> None:
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

        # Convert np.ndarray's dtype to their 32 counterparts
        data = [arr.astype(arr.dtype.kind) for arr in data]

        # Check for same num of examples in the multiple ndarray's
        axis = 0 if P.data_format() == "batch_first" else -1
        assert all(data[0].shape[axis] == arr.shape[axis] for arr in data), \
            (f"Unequal num of examples in {P.data_format()} (axis={axis}): "
             f"{[arr.shape for arr in data]} - is the data format correct?")

        dataset_dict = {f"arr_{idx}": arr for idx, arr in enumerate(data)}
        dataset_dict["__keys__"] = [f'arr_{idx}' for idx in range(len(data))]
        dataset_dict["saved_axis"] = axis

        # Save each ndarray (denoted by its key) seperately
        np.savez_compressed(path, **dataset_dict) if compress \
            else np.savez(path, **dataset_dict)


    @staticmethod
    def load(path: str, as_numpy: bool = False) -> Union[np.ndarray, \
            List[np.ndarray], TFDataset, TorchDataset]:
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
            out_axis = 0 if P.data_format() == "batch_first" else -1
            with np.load(path, allow_pickle=False) as npzfile:
                saved_axis = int(npzfile["saved_axis"])
                data = [np.moveaxis(npzfile[key], saved_axis, out_axis)
                        for key in npzfile["__keys__"]]
            return data[0] if len(data) == 1 else data
        return TorchNumpyDataset(path) if P.backend() == "pytorch" \
            else TensorflowNumpyDataset(path)


class TFRecordsSerializer(LazySerializer):
    """Serialize np.ndarray's to bytes and write to TFRecords file.

    Note that TFRecords does not support random access and is in fact
    not very performant. It's better to use :class:`LMDBSerializer`.
    """

    @staticmethod
    def save(data: Union[np.ndarray, List[np.ndarray]], path: str,
             save_index: bool = True) -> None:
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
            The ndarray's to serialize.

        path: str
            Output TFRecords file.

        save_index: bool, default=True
            If True, saves an index of records. If False, no index is
            created.
        """
        def _bytes_feature(value: Union[str, bytes]) -> tf.train.Feature:
            """Returns a bytes_list from a string / byte."""
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy() # BytesList won't unpack string from an EagerTensor.
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _float_feature(value: float) -> tf.train.Feature:
            """Returns a float_list from a float / double."""
            if not isinstance(value, (list, np.ndarray)):
                value = [value] # FloatList won't unpack unless it is an list/np.array.
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))

        def _int64_feature(value: Union[bool, int]) -> tf.train.Feature:
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

        def _create_idx(tfrecord_file: path, index_file: path) -> None:
            """Create index of TFRecords file. See https://git.io/JvGMw.

            The rows (contained within the file) indicates the num of
            examples in the dataset. The last column indicates how many
            bytes of storage each example takes.
            """
            infile = open(tfrecord_file, "rb")
            outfile = open(index_file, "w")

            while True:
                current = infile.tell()
                try:
                    byte_len = infile.read(8)
                    if len(byte_len) == 0:
                        break
                    infile.read(4)
                    proto_len = struct.unpack("q", byte_len)[0]
                    infile.read(proto_len)
                    infile.read(4)
                    outfile.write(str(current) + " " + str(infile.tell() - current) + "\n")
                except Exception:
                    print("Failed to parse TFRecord.")
                    break

            infile.close()
            outfile.close()

        if isinstance(data, np.ndarray):
            data = [data]

        # Convert np.ndarray's dtype to their 32 counterparts
        data = [arr.astype(arr.dtype.kind) for arr in data]

        # Check for same num of examples in the multiple ndarray's
        axis = 0 if P.data_format() == "batch_first" else -1
        assert all(data[0].shape[axis] == arr.shape[axis] for arr in data), \
            (f"Unequal num of examples in {P.data_format()} (axis={axis}): "
             f"{[arr.shape for arr in data]} - is the data format correct?")

        # Add shapes of each array in the dataset (for a single example). Hack
        # to allow serialized data to be reshaped properly when loaded.
        shapes = {f"shape_{idx}": np.delete(arr.shape, axis)
                  for idx, arr in enumerate(data)}
        dataset = {f"arr_{idx}": arr for idx, arr in enumerate(data)}
        dataset.update(shapes)

        # Write serialized example(s) into the dataset
        n_examples = data[0].shape[axis]
        with tf.io.TFRecordWriter(path) as writer:
            for row in tqdm(range(n_examples), total=n_examples):
                # NOTE: tobytes() flattens an ndarray. We have to flatten it
                # because tf _bytes_feature() only takes in bytes. To combat
                # this, we save each ndarray's shape as well (see above).
                example = {}
                for key, arr in dataset.items():
                    # Save metadata about the array (aka shape) as int64 feature
                    if key.startswith("shape"):
                        example[key] = {"data": arr, "_type": _int64_feature}
                    else:
                        example[key] = {"data": arr[row].tobytes() if P.data_format() \
                            == "batch_first" else arr[..., row].tobytes(),
                                        "_type": _bytes_feature}
                writer.write(_serialize(example))

        # Write index of the examples
        # NOTE: It's recommended to create an index file for each TFRecord file.
        # Index file must be provided when using multiple workers, otherwise the
        # loader may return duplicate records.
        if save_index:
            _create_idx(tfrecord_file=path, index_file=f"{path}_idx")


    @staticmethod
    def load(path: str, as_numpy: bool = False) -> Union[np.ndarray, \
            List[np.ndarray], TFDataset, TorchDataset]:
        """Load the dataset.

        Assumes that each record's np.ndarrays are saved under default
        names `arr_0`, ..., `arr_n` and its associated shapes as
        `shape_0`, ..., `shape_n`.

        Params:
        -------
        path: str
            TFRecords file which contains the dataset.

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
        # Parse serialized records into correctly shaped tensors/ndarray's
        if as_numpy:
            dataset_dict = {}
            axis = 0 if P.data_format() == "batch_first" else -1
            for serialized in tf.python_io.tf_record_iterator(path):
                example = tf.train.Example()
                example.ParseFromString(serialized)
                # Save array shapes to ensure respective ndarrays get reshaped properly
                shapes = {name: list(example.features.feature[name].int64_list.value)
                          for name in example.features.feature.keys() if name.startswith("shape")}
                for name, tf_feature in example.features.feature.items():
                    if name.startswith("arr"):
                        parsed_data = np.frombuffer(tf_feature.bytes_list.value[0], np.float)
                        newshape = shapes.get("shape_{}".format(name.split("_")[-1]))
                        newshape = [1] + newshape if P.data_format() == \
                            "batch_first" else newshape + [1]
                        reshaped = np.reshape(parsed_data, newshape=newshape)
                        # Concatenate individual examples together into one ndarray.
                        if name not in dataset_dict.keys():
                            dataset_dict[name] = reshaped
                        else:
                            dataset_dict[name] = np.concatenate((dataset_dict.get(name), \
                                reshaped), axis=axis)
            # Extract np.ndarray's from dict and return in its original form
            data = dataset_dict.values()
            return data[0] if len(data) == 1 else data
        return TorchTFRecordsDataset(path) if P.backend() == "pytorch" \
            else TFRecordsDataset(path)
        