"""
Modules to split datasets into train, validation, and test sets using a variety 
of splitting techniques. 
"""

from profit.dataset.splitter import base_splitter, random_splitter, stratified_splitter
from profit.dataset.splitter.base_splitter import BaseSplitter
from profit.dataset.splitter.random_splitter import RandomSplitter
from profit.dataset.splitter.stratified_splitter import StratifiedSplitter