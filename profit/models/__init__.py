"""Various models.

In general, the hope is all models will be implemented using both torch
and tensorflow backends, with the framework-specific logic lying in each
of their respective folders.
"""

from profit.models import tensorflow
from profit.models import torch
