from __future__ import absolute_import
from __future__ import print_function


# Type of format to use throughout the session.
_DATA_FORMAT = 'batch_first'


def data_format() -> str:
    """Returns the default data format convention.

    Returns:
    -------
    format: str
        A string, either `'batch_first'` or `'batch_last'`.

    Example:
    --------
    ```python
    >>> profit.backend.data_format()
    'batch_first'
    ```
    """
    return _DATA_FORMAT


def set_data_format(data_format: str):
    """Sets the value of the data format convention.

    Params:
    -------
    data_format: str
        Either `'batch_first'` or `'batch_last'`.

    Example:
    --------
    ```python
    >>> from profit import backend as P
    >>> P.data_format()
    'batch_first'
    >>> P.set_data_format('batch_last')
    >>> P.data_format()
    'batch_last'
    ```
    """
    global _DATA_FORMAT
    if data_format not in {'batch_last', 'batch_first'}:
        raise ValueError('Unknown data_format:', data_format)
    _DATA_FORMAT = str(data_format)