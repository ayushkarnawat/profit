"""
Modules to split datasets into train, validation, and test sets using a variety 
of splitting techniques. 
"""

from profit.dataset.splitters import base_splitter, random_splitter, stratified_splitter
from profit.dataset.splitters.base_splitter import BaseSplitter
from profit.dataset.splitters.random_splitter import RandomSplitter
from profit.dataset.splitters.stratified_splitter import StratifiedSplitter