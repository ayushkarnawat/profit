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


class TFRecordsDataset(object):
    """Parse (generic) TFRecords file into a `tf.data.Dataset` object, 
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


class TorchTFRecordsDataset(Dataset):
    """Parse (generic) tensorflow dataset into a `torch.utils.data.Dataset` 
    object, which contains `torch.Tensor`s.
    
    Ideally, this should be done without need for using tensorflow. See: 
    https://discuss.pytorch.org/t/read-dataset-from-tfrecord-format/16409

    This seems promising, however requires using another pkg: 
    https://github.com/vahidk/tfrecord. Plus, we have to modify to work with np.ndarrays.

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
    # from torch.utils.data import DataLoader
    # torch_dataset = TorchHDF5Dataset("data/3gb1/processed/transformer_fitness/primary5.h5")
    # torch_dataset = TorchLMDBDataset("data/3gb1/processed/transformer_fitness/primary.mdb")
    # torch_dataset = TorchNumpyDataset("data/3gb1/processed/transformer_fitness/primary5.npz")
    # torch_dataset = TorchTFRecordsDataset("data/3gb1/processed/transformer_fitness/primary.tfrecords")
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
        print(dataset)
    # print([arr.shape for arr in dataset])
    # print(isinstance(dataset, torch.utils.data.Dataset))
    # loader = DataLoader(dataset, batch_size=1)
    # for i_batch, sample_batched in enumerate(loader):
    #     print(i_batch, [arr.shape for arr in sample_batched])
    # # Transpose to make it channels_last
    # dataset = [arr.T for arr in dataset]
    # print([arr.shape for arr in dataset])

    # P.set_data_format('channels_last')
    # serializer = serialize_method_dict.get('npz')
    # serializer.save(dataset, path="test.npz")

    # torch_dataset = TorchNumpyDataset('test.npz')
    # loader = DataLoader(torch_dataset, batch_size=2)
    # for i_batch, sample_batched in enumerate(loader):
    #     print(i_batch, [arr.shape for arr in sample_batched])
    # for i in range(len(torch_dataset)): 
    #     sample = torch_dataset[i]
    #     print(i, [tensor.shape for tensor in sample])