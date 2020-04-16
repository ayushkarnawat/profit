import os
from typing import List, Union

import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from profit.dataset import generator
from profit.dataset.parsers.data_frame_parser import DataFrameParser
from profit.dataset.preprocessing.mutator import PDBMutator
from profit.dataset.preprocessors import preprocess_method_dict
from profit.utils.data_utils import serialize_method_dict
from profit.utils.data_utils.cacher import CacheNamePolicy


def load_dataset(method, mutator_fmt, labels, rootdir='data/3gb1/processed/',
                 num_data=-1, filetype='mdb', as_numpy=False, **pp_kwargs
                 ) -> Union[np.ndarray, List[np.ndarray], Dataset]:
    """Load pre-processed dataset.

    If we are loading a large database, it does not make sense to load
    the data into memory. Rather, we just pass the location of the saved
    filepath.

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
    data: np.ndarray, list of np.ndarray, or torch.utils.data.Dataset
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
        print(f"Preprocessing dataset using {type(preprocessor).__name__}...")

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
        print(f"Serializing dataset using {type(serializer).__name__}...")
        serializer.save(data=data, path=data_path)

    # Load data from cache
    print(f"Loading preprocessed data from cache `{data_path}`")
    data = serializer.load(path=data_path, as_numpy=as_numpy)
    return data


def load_variants(method, labels, rootdir='data/3gb1/processed/',
                  num_data=-1, filetype='mdb', as_numpy=False, **pp_kwargs
                  ) -> Union[np.ndarray, List[np.ndarray], Dataset]:
    """Generate and load (sequence) variants.

    NOTE: For now, this only works with sequence-based preprocessors.
    """
    policy = CacheNamePolicy(method, "variants", labels, rootdir=rootdir,
                             num_data=num_data, filetype=filetype, **pp_kwargs)
    data_path = policy.get_data_file_path()
    serializer = serialize_method_dict.get(filetype)()

    # Compute features
    if not os.path.exists(data_path):
        # Initalize class(es)
        preprocessor = preprocess_method_dict.get(method)(**pp_kwargs)
        print(f"Preprocessing dataset using {type(preprocessor).__name__}...")

        # Generate (all) variants
        template = list("MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE")
        pos = [39, 40, 41, 54]
        num_mutation_sites = len(pos)
        seqs = generator.gen(n=num_mutation_sites)

        # Select `num_data` random sequences, if we are only taking a subset
        if num_data > 0:
            random_state = 1
            np.random.seed(random_state)
            idx = np.random.choice(len(seqs), size=num_data, replace=False)
            seqs = seqs[idx]

        # Compute features
        features = []
        for seq in seqs:
            if len(seq) != len(pos):
                raise ValueError(f"The positions to change from the sequence "
                                 f"`{seq}` (N={len(seq)}) does not match the "
                                 f"positions (N={len(pos)}) specified.")
            for idx, aa in zip(pos, seq):
                template[idx-1] = aa
            features.append(preprocessor.get_input_feats(template))
        features = np.array(features)

        # Cache results using default array names
        policy.create_cache_directory()
        print(f"Serializing dataset using {type(serializer).__name__}...")
        serializer.save(data=features, path=data_path)

    # Load data from cache
    print(f"Loading preprocessed data from cache `{data_path}`")
    data = serializer.load(path=data_path, as_numpy=as_numpy)
    return data
