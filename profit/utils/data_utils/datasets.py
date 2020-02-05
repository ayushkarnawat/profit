import os
import json

from functools import partial
from typing import Dict, List, Tuple, Union

import h5py
import lmdb
import numpy as np
import pickle as pkl
import tensorflow as tf

import torch
from torch.utils.data import Dataset, IterableDataset

from tfreader import tfrecord_iterator, tfrecord_loader


def TensorflowHDF5Dataset(path: str) -> tf.data.Dataset:
    """Parse (generic) HDF5 dataset into a `tf.data.Dataset` object, 
    which contains `tf.Tensor`s.

    Params:
    -------
    path: str
        HDF5 file which contains dataset.
    
    Returns:
    --------
    dataset: tf.data.Dataset
        Dataset loaded into its `tf.data.Dataset` form.
    """
    class HDF5DatasetGenerator(object):
        """Generates dataset examples contained in a .h5/.hdf5 file. 
        
        NOTE: We pass this object into `tf.data.Dataset.from_generator()` 
        to load the `tf.data.Dataset` object.
        """

        def __init__(self, path: str):
            self.h5file = h5py.File(path, "r")
            self.keys = list(self.h5file.keys())

            # Retrive first sample to help recover proper types/shapes of ndarrays. 
            # NOTE: We do not check if all the samples have the same type/shape
            # since (a) we checked for it when saving the file and (b) it is more 
            # efficient. However, if the file is modified in between saving and 
            # loading (see https://w.wiki/GQE), this could lead to potential issues. 
            # TODO: Do we check if all the samples have the same type/shape?
            self.example = json.loads(self.h5file.get(self.keys[0])[()])

        def __call__(self):
            # Yield a dict with "arr_n" as the key and the ndarray as the value
            for key in self.keys:
                yield json.loads(self.h5file.get(key)[()])

        @property
        def output_shapes(self):
            """Defines the data shapes used to store the dataset. 
            
            Helps recover `np.ndarray`s types when loading into a 
            `tf.data.Dataset` object using `from_generator()`.
            """
            return {key: np.array(arr).shape for key, arr in self.example.items()}

        @property
        def output_types(self):
            """Defines the data types used to store the dataset. 
            
            Helps recover `np.ndarray`s shapes when loading into a 
            `tf.data.Dataset` object using `from_generator()`.
            """
            return {key: tf.float32 for key in self.example.keys()}

    dsg = HDF5DatasetGenerator(path)
    return tf.data.Dataset.from_generator(dsg, dsg.output_types, dsg.output_shapes)


def TensorflowLMDBDataset(path: str) -> tf.data.Dataset:
    """Parse (generic) LMDB dataset into a `tf.data.Dataset` object, 
    which contains `tf.Tensor`s.

    Params:
    -------
    path: str
        LMDB file which contains dataset.

    Returns:
    --------
    dataset: tf.data.Dataset
        Dataset loaded into its `tf.data.Dataset` form.
    """
    class LMDBDatasetGenerator(object):
        """Generates dataset examples contained in a .mdb/.lmdb file. 
        
        NOTE: We pass this object into `tf.data.Dataset.from_generator()` 
        to load the `tf.data.Dataset` object.
        """

        def __init__(self, path: str):
            # Check whether directory or full filename is provided. If dir, 
            # check for "data.mdb" file within dir.
            isdir = os.path.isdir(path)
            if isdir:
                default_path = os.path.join(path, "data.mdb")
                assert os.path.isfile(default_path), "LMDB default file {} " \
                    "does not exist!".format(default_path)
            else:
                assert os.path.isfile(path), f"LMDB file {path} does not exist!"

            self.db = lmdb.open(path, subdir=isdir, readonly=True, readahead=False)

            # Retrive first sample to help recover proper types/shapes of ndarrays. 
            # NOTE: We do not check if all the samples have the same type/shape
            # since (a) we checked for it when saving the file and (b) it is more 
            # efficient. However, if the file is modified in between saving and 
            # loading (see https://w.wiki/GQE), this could lead to potential issues. 
            # TODO: Do we check if all the samples have the same type/shape?
            with self.db.begin() as txn, txn.cursor() as cursor:
                self.keys = pkl.loads(cursor.get(b"__keys__"))
                self.example = {f"arr_{i}": arr for i,arr in enumerate(\
                    pkl.loads(cursor.get(self.keys[0])))}

        def __call__(self):
            # Yield a dict with "arr_n" as the key and the ndarray as the value
            with self.db.begin() as txn, txn.cursor() as cursor:
                for key in self.keys:
                    example = pkl.loads(cursor.get(key))
                    yield {f"arr_{i}":arr for i,arr in enumerate(example)}

        @property
        def output_shapes(self):
            """Defines the data shapes used to store the dataset.

            Helps recover `np.ndarray`s types when loading into a 
            `tf.data.Dataset` object using `from_generator()`.
            """
            return {key: arr.shape for key, arr in self.example.items()}

        @property
        def output_types(self):
            """Defines the data types used to store the dataset. 
            
            Helps recover `np.ndarray`s shapes when loading into a 
            `tf.data.Dataset` object using `from_generator()`.
            """
            return {key: tf.float32 for key in self.example.keys()}

    dsg = LMDBDatasetGenerator(path)
    return tf.data.Dataset.from_generator(dsg, dsg.output_types, dsg.output_shapes)


def TensorflowNumpyDataset(path: str) -> tf.data.Dataset:
    """Parse (generic) numpy dataset into a `tf.data.Dataset` object, 
    which contains `tf.Tensor`s.

    Params:
    -------
    path: str
        Npz file which contains dataset.

    Returns:
    --------
    dataset: tf.data.Dataset
        Dataset loaded into its `tf.data.Dataset` form.
    """
    class NumpyDatasetGenerator(object):
        """Generates dataset examples contained in a .npz file. 
        
        NOTE: We pass this object into `tf.data.Dataset.from_generator()` 
        to load the `tf.data.Dataset` object.
        """

        def __init__(self, path: str):
            self.npzfile = np.load(path, allow_pickle=False)
            self.keys = list(self.npzfile.keys())

            # Retrive first sample to help recover proper types/shapes of ndarrays. 
            # NOTE: We do not check if all the samples have the same type/shape
            # since (a) we checked for it when saving the file and (b) it is more 
            # efficient. However, if the file is modified in between saving and 
            # loading (see https://w.wiki/GQE), this could lead to potential issues. 
            # TODO: Do we check if all the samples have the same type/shape?
            self.example = pkl.loads(self.npzfile.get(self.keys[0]))

        def __call__(self):
            # Yield a dict with "arr_n" as the key and the ndarray as the value
            for key in self.keys:
                yield pkl.loads(self.npzfile.get(key))

        @property
        def output_shapes(self):
            """Defines the data shapes used to store the dataset. 
            
            Helps recover `np.ndarray`s types when loading into a 
            `tf.data.Dataset` object using `from_generator()`.
            """
            return {key: np.array(arr).shape for key, arr in self.example.items()}

        @property
        def output_types(self):
            """Defines the data types used to store the dataset. 
            
            Helps recover `np.ndarray`s shapes when loading into a 
            `tf.data.Dataset` object using `from_generator()`.
            """
            return {key: tf.float32 for key in self.example.keys()}

    dsg = NumpyDatasetGenerator(path)
    return tf.data.Dataset.from_generator(dsg, dsg.output_types, dsg.output_shapes)


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
                     **kwargs: Dict[str, List[int]]) -> Tuple[tf.Tensor, ...]:
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

    NOTE: Regardless of the format the data is saved in, the examples 
    are always concatenated vertically row-wise (aka first channel).

    Params:
    -------
    path: str
        HDF5 file which contains dataset.
    """

    def __init__(self, path: str):
        self.h5file = h5py.File(path, "r")
        self.keys = list(self.h5file.keys())

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, List[torch.Tensor]]:
        # Convert to pytorch tensors
        sample_dict = json.loads(self.h5file.get(self.keys[idx])[()])
        sample = [torch.FloatTensor(arr) for arr in sample_dict.values()]
        return sample[0] if len(sample) == 1 else sample


class TorchLMDBDataset(Dataset):
    """Parse (generic) LMDB dataset into a `torch.utils.data.Dataset` 
    object, which contains `torch.Tensor`s.
    
    NOTE: Regardless of the format the data is saved in, the examples 
    are always concatenated vertically row-wise (aka first channel).

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

    NOTE: Regardless of the format the data is saved in, the examples 
    are always concatenated vertically row-wise (aka first channel).

    Params:
    -------
    path: str
        Npz file which contains dataset.
    """

    def __init__(self, path: str):
        self.npzfile = np.load(path, allow_pickle=False)
        self.keys = list(self.npzfile.keys())

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, List[torch.Tensor]]:
        # Convert to pytorch tensors
        sample_dict = pkl.loads(self.npzfile.get(self.keys[idx]))
        sample = [torch.FloatTensor(arr) for arr in sample_dict.values()]
        return sample[0] if len(sample) == 1 else sample


class TorchTFRecordsDataset(IterableDataset):
    """Parse (generic) tensorflow dataset into a `torch.utils.data.IterableDataset` 
    object, which contains `torch.Tensor`s.
    
    NOTE: This requires using tensorflow (via `tfrecord_loader()`). As 
    such, the whole purpose of loading data from a tfrecords file to 
    `torch.Tensor`s is defeated as we should be able to do without 
    using the tf package. 

    As shown in https://github.com/vahidk/tfrecord, we don't have to 
    use `tf.train.Example()` to parse through examples. Rather, we can 
    use `example_pb2.py` provided in the link above to parse an example 
    by replacing `tf.train.Example()` -> `example_pb2.Example()`.  

    Params:
    -------
    path: str
        TFRecords file which contains dataset.
    """

    def __init__(self, path: str):
        self.tfrecord_path = path
        self.index_path = f"{path}_idx" if os.path.exists(f"{path}_idx") else None

        # Retrieve the keys and their data types from the first example to help 
        # recover proper shapes of the ndarrays. 
        serialized = next(tfrecord_iterator(self.tfrecord_path))
        example = tf.train.Example()
        example.ParseFromString(serialized)
        feature_keys = list(example.features.feature.keys())
        self.description = {key: "int" if key.startswith("shape") else "byte" 
                            for key in feature_keys}

        if self.index_path == None:
            print('Could not find index path - sharding and num-workers will be unavailable!')

    def __iter__(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        # Shard dataset if many workers are iterating over dataset
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            shard = worker_info.id, worker_info.num_workers
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
        else:
            shard = None

        # Reshape example into correctly shaped tensors/ndarray's
        it = tfrecord_loader(self.tfrecord_path, self.index_path, self.description, shard)
        for record in it:
            sample = []
            for key, value in record.items():
                if key.startswith("arr"):
                    shape = record.get("shape_{}".format(key.split("_")[-1]))
                    sample.append(np.reshape(value, newshape=shape))
            yield sample[0] if len(sample) == 1 else sample


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    # torch_dataset = TorchHDF5Dataset("data/3gb1/processed/transformer_fitness/primary5.h5")
    # torch_dataset = TorchLMDBDataset("data/3gb1/processed/transformer_fitness/primary5.mdb")
    # torch_dataset = TorchNumpyDataset("data/3gb1/processed/transformer_fitness/primary5.npz")
    # torch_dataset = TorchTFRecordsDataset("data/3gb1/processed/transformer_fitness/primary5.tfrecords")
    # sample = torch_dataset[755]
    # for i in range(len(torch_dataset)): 
    #     sample = torch_dataset[i]
    #     print(i, [tensor.shape for tensor in sample])
    # loader = DataLoader(dataset, batch_size=2)
    # for i_batch, sample_batched in enumerate(loader):
    #     print(i_batch, [arr.shape for arr in sample_batched])

    # Load the dataset
    import os
    from typing import List, Union

    import numpy as np
    import pandas as pd

    from profit.dataset.parsers.data_frame_parser import DataFrameParser
    from profit.dataset.preprocessing.mutator import PDBMutator
    from profit.dataset.preprocessors import preprocess_method_dict
    from profit.utils.data_utils import serialize_method_dict
    from profit.utils.data_utils.cacher import CacheNamePolicy
    from profit import backend as P

    def load_dataset(method, mutator_fmt, labels, rootdir='data/3gb1/processed/', 
                    num_data=-1, filetype='h5', as_numpy=False) -> Union[np.ndarray, List[np.ndarray]]:
        """Load pre-processed dataset.

        Returns:
        --------
        data: list of np.ndarray (multiple) or np.ndarray (single) or str
            List of np.ndarray if filetype saved in 'npy', 'npz', 'h5', or 
            'hdf5' format. If saved into a database (e.g. tfrecords or 
            lmdb), the filepath where database is saved is returned.
        """
        policy = CacheNamePolicy(method, mutator_fmt, labels, rootdir=rootdir, 
                                num_data=num_data, filetype=filetype)
        data_path = policy.get_data_file_path()
        serializer = serialize_method_dict.get(filetype)()

        # Compute features
        if not os.path.exists(data_path):
            # Initalize class(es)
            preprocessor = preprocess_method_dict.get(method)()
            mutator = PDBMutator(fmt=mutator_fmt) if mutator_fmt else None
            print('Preprocessing dataset using {}...'.format(type(preprocessor).__name__))

            # Preprocess dataset w/ mutations if requested
            target_index = np.arange(num_data) if num_data >= 0 else None
            df = pd.read_csv('data/3gb1/raw/fitness570.csv', sep=',')
            if 'PDBID' not in list(df.columns):
                df['PDBID'] = ['3gb1' for _ in range(len(df))] 
            if 'Positions' not in list(df.columns):
                df['Positions'] = [[39, 40, 41, 54] for _ in range(len(df))]
            parser = DataFrameParser(preprocessor, mutator, data_col='Variants', \
                pdb_col='PDBID', pos_col='Positions', labels=labels, \
                process_as_seq=True)
            data = parser.parse(df, target_index=target_index)['dataset']

            # Cache results using default array names
            policy.create_cache_directory()
            print('Serializing dataset using {}...'.format(type(serializer).__name__))
            serializer.save(data=data, path=data_path)
        
        # Load data from cache
        print('Loading preprocessed data from cache `{}`'.format(data_path))
        data = serializer.load(path=data_path, as_numpy=as_numpy)
        return data

    for ftype in ['h5', 'mdb', 'npz', 'tfrecords']:
        dataset = load_dataset('transformer', 'primary', labels='Fitness', num_data=5, \
            filetype=ftype, as_numpy=False)
        
        # Print data shapes if torch datasets
        if isinstance(dataset, torch.utils.data.Dataset):
            loader = DataLoader(dataset, batch_size=2)
            for i_batch, sample_batched in enumerate(loader):
                print(i_batch, [arr.shape for arr in sample_batched])
        elif isinstance(dataset, tf.data.Dataset):
            print(dataset)
        elif isinstance(dataset, (list, np.ndarray)):
            print([arr.shape for arr in dataset])
    
    # # Transpose to make it channels_last
    # dataset = [arr.T for arr in dataset]
    # print([arr.shape for arr in dataset])

    # P.set_data_format('channels_last')
    # serializer = serialize_method_dict.get('npz')
    # serializer.save(dataset, path="test.npz")