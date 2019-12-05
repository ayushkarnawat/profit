import os

import numpy as np
import pandas as pd

from profit.dataset.parsers.data_frame_parser import DataFrameParser
from profit.dataset.preprocessing.mutator import PDBMutator
from profit.dataset.preprocessors import preprocess_method_dict
from profit.utils.data_utils.cacher import CacheNamePolicy


def load_dataset(method, mutator_fmt, labels, rootdir='data/3gb1/processed/', num_data=-1, out_size=50) -> np.ndarray:
    """
    TODO: Change to variable filepath. Should we allow for hardcoded parser? 
    Depends on variable filepath extension. As of now, we have hardcoded this 
    info because the filepath is known. 
    TODO: Allow user to specify columns names for the PDB id and positions. 
    As of now, they are hardcoded to 'PDBID' and 'Positions'
    TODO: Specify max_size for proper feature concatenation. useful in this 
    scenario, because we cannot specify the larget size we want to use.    
    """
    policy = CacheNamePolicy(method, mutator_fmt, labels, rootdir=rootdir, num_data=num_data)
    data_path = policy.get_data_file_path()

    # Load data from cache, if it exists
    data = None
    if os.path.exists(data_path):
        print('Loading preprocessed data from cache `{}`'.format(data_path))
        npzdata = np.load(data_path)
        data = list(dict(npzdata).values())
    if data is None:
        # Initalize class(es)
        preprocessor = preprocess_method_dict.get(method)(out_size=out_size)
        mutator = PDBMutator(fmt=mutator_fmt, remove_tmp_file=False)
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

        # Cache dataset using default array names
        policy.create_cache_directory()
        np.savez(data_path, *data)
    return data


if __name__ == "__main__":
    data = load_dataset('gcn', 'tertiary', labels='Fitness', num_data=10, out_size=1000)
    for arr in data:
        print(arr.shape)