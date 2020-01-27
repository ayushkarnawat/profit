"""
TODO: Fix "Using TensorFlow backend." when importing `profit` module. 
This confuses the user, as the default backend is the pytorch one.
"""

from profit.backend import common
from profit.backend import load_backend

from profit.backend.common import data_format
from profit.backend.common import set_data_format
from profit.backend.load_backend import backend