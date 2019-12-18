import os
from typing import Optional
from profit.utils.io_utils import maybe_create_dir


class CacheNamePolicy(object):
    """Cached dataset naming policy.
    
    Contains filepaths associated with the processed data.

    Params:
    -------
    method: str
        The model applied to the dataset.
    
    mutator: str or None, optional, default=None
        Mutator format type (primary or tertiary) applied when 
        preprocessing the data. If None or '', it signifies that no 
        mutageneis was performed.

    labels: str or None, optional, default=None
        Label processed. If None or '', then all labels (for multi-task 
        learning) were used. 

    rootdir: str, default='base'
        Base directory where all cached data is stored.

    num_data: int, default=-1
        The number of datapoints saved in the dataset. If -1, then all 
        datapoints were processed.

    filetype: str, default="h5"
        What extension to save the data with. The supported types are 
        'h5', 'hdf5', 'lmdb', 'mdb', 'npz', 'tfrecords'.
    """

    def __init__(self, method: str, mutator: Optional[str]=None, 
                 labels: Optional[str]=None, rootdir: str='base', 
                 num_data: int=-1, filetype: str="h5") -> None:
        assert filetype in ['h5', 'hdf5', 'lmdb', 'mdb', 'npz', 'tfrecords']
        num_data_str = "{0:d}".format(num_data) if num_data >= 0 else ""
        mutator_str = "{0:s}".format(mutator.lower()) if mutator else "original"
        self.filename = "{}{}.{}".format(mutator_str, num_data_str, filetype)
        self.method = method
        self.labels = labels
        self.rootdir = rootdir
        self.cache_dir = self._get_cache_directory_path(rootdir, method, \
            labels)


    def _get_cache_directory_path(self, rootdir: str, method: str, 
                                  labels: Optional[str]=None) -> str:
        """Get cache directory where data is stored.
        
        Params:
        -------
        rootdir: str,
            Base directory where all cached data is stored.
        
        method: str
            The model applied to the dataset.

        labels: str or None, optional, default=None
            Label processed. If None or '', then all labels (for multi-task 
            learning) were used. 

        Returns:
        --------
        dir_path: str
            Cache directory path.
        """
        labels_str = '_{}'.format(labels.lower()) if labels else '_all'
        return os.path.join(rootdir, '{}{}'.format(method.lower(), labels_str))


    def get_data_file_path(self) -> str:
        """Get the full filepath to where data is going to be cached.
        
        Returns:
        --------
        path: str
            Full path where cached file is located (w/in the directory).
        """
        return os.path.join(self.cache_dir, self.filename)


    def create_cache_directory(self) -> None:
        """Creates cache directory."""
        return maybe_create_dir(self.cache_dir)