"""Setup the default profit backend. Allows choice of using pytorch or
keras/tensorflow models. Additionally, specifies how the data is
formatted (for generic lazy loading and batching examples).

NOTE: We do not have a `set_backend()` function as, if the user decides
to change the backend in the middle of the script/program, that would
mess up the computation. To prevent such an occurence, it is RECOMMENED
that the user modify the `backend` key within `~/.profit.profit.json`.
This will (re)load the specified backend before any computations have
been performed.

Adapted from the keras backend: https://git.io/JvqPB.
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import json

from profit.backend.common import data_format
from profit.backend.common import set_data_format


# Set profit base dir path given PROFIT_HOME env variable, if applicable.
# Otherwise either ~/.profit or /tmp.
if 'PROFIT_HOME' in os.environ:
    _profit_dir = os.environ.get('PROFIT_HOME')
else:
    _profit_base_dir = os.path.expanduser('~')
    if not os.access(_profit_base_dir, os.W_OK):
        _profit_base_dir = '/tmp'
    _profit_dir = os.path.join(_profit_base_dir, '.profit')

# Default backend: Torch.
_BACKEND = 'torch'

# Attempt to read profit config file.
_config_path = os.path.expanduser(os.path.join(_profit_dir, 'profit.json'))
if os.path.exists(_config_path):
    try:
        with open(_config_path) as f:
            _config = json.load(f)
    except ValueError:
        _config = {}
    _backend = _config.get('backend', _BACKEND)
    assert _backend in {'torch', 'tensorflow'}
    _data_format = _config.get('data_format', data_format())
    assert _data_format in {'batch_last', 'batch_first'}

    set_data_format(_data_format)
    _BACKEND = _backend

# Save config file, if possible.
if not os.path.exists(_profit_dir):
    try:
        os.makedirs(_profit_dir)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass

if not os.path.exists(_config_path):
    # Use default params
    _config = {
        'backend': _BACKEND,
        'data_format': data_format()
    }
    try:
        with open(_config_path, 'w') as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:
        # Except permission denied.
        pass

# Set backend based on PROFIT_BACKEND flag, if applicable.
if 'PROFIT_BACKEND' in os.environ:
    _backend = os.environ['PROFIT_BACKEND']
    if _backend:
        _BACKEND = _backend


def backend() -> str:
    """Returns the name of the current backend (e.g. "torch").

    Returns:
    --------
    backend: str
        The name of the backend profit is currently using.

    Example:
    --------
    ```python
    >>> profit.backend.backend()
    'torch'
    ```
    """
    return _BACKEND
