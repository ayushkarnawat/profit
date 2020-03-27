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
                 num_data=-1, filetype='h5', as_numpy=False, **pp_kwargs
                 ) -> Union[np.ndarray, List[np.ndarray]]:
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
                             num_data=num_data, filetype=filetype, **pp_kwargs)
    data_path = policy.get_data_file_path()
    serializer = serialize_method_dict.get(filetype)()

    # Compute features
    if not os.path.exists(data_path):
        # Initalize class(es)
        preprocessor = preprocess_method_dict.get(method)(**pp_kwargs)
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
