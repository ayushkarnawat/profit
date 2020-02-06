from __future__ import absolute_import
from __future__ import print_function


# Type of float to use throughout the session.
_DATA_FORMAT = 'channels_first'


def data_format() -> str:
    """Returns the default data format convention.

    Returns:
    -------
    format: str
        A string, either `'channels_first'` or `'channels_last'`.

    Example:
    --------
    ```python
    >>> profit.backend.data_format()
    'channels_first'
    ```
    """
    return _DATA_FORMAT


def set_data_format(data_format: str):
    """Sets the value of the data format convention.

    Params:
    -------
    data_format: str
        Either `'channels_first'` or `'channels_last'`.

    Example:
    --------
    ```python
    >>> from profit import backend as P
    >>> P.data_format()
    'channels_first'
    >>> P.set_data_format('channels_last')
    >>> P.data_format()
    'channels_last'
    ```
    """
    global _DATA_FORMAT
    if data_format not in {'channels_last', 'channels_first'}:
        raise ValueError('Unknown data_format:', data_format)
    _DATA_FORMAT = str(data_format)