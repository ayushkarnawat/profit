"""Split datasets into train, val, and test sets using different
splitting techniques.
"""

from profit.dataset.splitters import base_splitter
from profit.dataset.splitters import random_splitter
from profit.dataset.splitters import stratified_splitter

from profit.dataset.splitters.base_splitter import BaseSplitter
from profit.dataset.splitters.random_splitter import RandomSplitter
from profit.dataset.splitters.stratified_splitter import StratifiedSplitter

split_method_dict = {
    'random': RandomSplitter,
    'stratified': StratifiedSplitter
}
