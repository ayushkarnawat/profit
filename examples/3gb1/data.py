import os
import warnings

import h5py
import numpy as np
import pandas as pd

from profit.dataset.parsers.data_frame_parser import DataFrameParser
from profit.dataset.preprocessing.mutator import PDBMutator
from profit.dataset.preprocessors import preprocess_method_dict
from profit.utils.data_utils.cacher import CacheNamePolicy
from profit.utils.data_utils.serializer import dataset_to_tfrecords


def load_dataset(method, mutator_fmt, labels, rootdir='data/3gb1/processed/', 
                 num_data=-1, filetype='h5') -> np.ndarray:
    """Load pre-processed dataset.

    If we are loading a large database, it does not make sense to load 
    the data into memory. Rather, we just pass the location of the 
    saved filepath.

    TODO: Change to variable filepath. Should we allow for hardcoded 
    parser? Depends on variable filepath extension. As of now, we have 
    hardcoded this info because the filepath is known. 

    TODO: Allow user to specify columns names for the PDB id and 
    positions. As of now, they are hardcoded to 'PDBID' and 'Positions'.

    TODO: Add MemoryError() if the full dataset cannot be loaded into 
    memory? Only useful so that the script does not load a arge dataset 
    into memory and slow the training down. If we can load the full 
    dataset into memory without any major issues, do so. Otherwise, 
    raise a MemoryError("Cannot load full dataset, unavailable memory.")

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
        if filetype == "npy":
            if len(data) > 1:
                warnings.warn("Unable to save {} ndarray's into .npy file. " \
                    "Use another filetype to cache results.".format(len(data)))
            else:
                np.save(data_path, data)
        elif filetype == "npz":
            np.savez_compressed(data_path, *data)
        elif filetype == "h5" or filetype == ".hdf5":
            h5file = h5py.File(data_path, "w")
            for idx, arr in enumerate(data):
                name = "arr_{}".format(idx)
                h5file.create_dataset(name, data=arr, chunks=True, compression="gzip")
            h5file.close()
        elif filetype == "tfrecords":
            dataset_to_tfrecords(data, data_path)
        elif filetype == "lmdb":
            raise NotImplementedError
        else:
            raise NotImplementedError("{} filetype not currently supported.".\
                format(filetype))

    # If filetype is a serialized database (e.g. tfrecords or lmdb)
    if filetype == "tfrecords" or filetype == "lmdb":
        warnings.warn("Not loading dataset into memory. Returning path " \
            "to saved database instead.")
        data = data_path
    
    # Load data from cache, if it exists
    if data is None:
        print('Loading preprocessed data from cache `{}`'.format(data_path))
        if filetype == "npy":
            data = np.load(data_path, allow_pickle=False)
        elif filetype == "npz":
            with np.load(data_path, allow_pickle=False) as npzfile:
                data = [arr[:] for arr in list(npzfile.values())]
        elif filetype == "h5" or filetype == "hdf5":
            with h5py.File(data_path, "r") as h5file:
                data = [h5file.get(key)[:] for key in list(h5file.keys())]
    return data