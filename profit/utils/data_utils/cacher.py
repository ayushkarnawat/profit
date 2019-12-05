import os
from typing import Optional


class CacheNamePolicy(object):
    """Cached dataset naming policy.
    
    Contains filepaths associated with the processed data.
    TODO: Change whole filenames to lowercase?

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

    rootdir: str, default='input'
        Base directory where all cached data is stored.

    num_data: int, default=-1
        The number of datapoints saved in the dataset. If -1, then all 
        datapoints were processed.
    """

    def __init__(self, method: str, mutator: Optional[str]=None, 
                 labels: Optional[str]=None, rootdir: str='input', 
                 num_data: int=-1) -> None:
        self.filename = '{}.npz'.format(mutator) if mutator else 'original.npz'
        self.method = method
        self.labels = labels
        self.rootdir = rootdir
        self.num_data = num_data
        self.cache_dir = self._get_cache_directory_path(rootdir, method, \
            labels, num_data)


    def _get_cache_directory_path(self, rootdir: str, method: str, 
                                  labels: Optional[str]=None, 
                                  num_data: int=-1) -> str:
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

        num_data: int, default=-1
            The number of datapoints saved in the dataset. If -1, then all 
            datapoints were processed.

        Returns:
        --------
        dir_path: str
            Cache directory path.
        """
        labels_str = '_{}'.format(labels) if labels else '_all'
        num_data_str = '{}'.format(num_data) if num_data >= 0 else ''
        return os.path.join(rootdir, '{}{}{}'.format(method, labels_str, num_data_str))


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
        try:
            os.makedirs(self.cache_dir)
        except OSError:
            if not os.path.isdir(self.cache_dir):
                raise