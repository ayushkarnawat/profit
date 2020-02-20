"""Module for feature generation and extraction."""

from profit.dataset.preprocessors import base_preprocessor
from profit.dataset.preprocessors import egcn_preprocessor
from profit.dataset.preprocessors import mol_preprocessor
from profit.dataset.preprocessors import seq_preprocessor
from profit.dataset.preprocessors import transformer_preprocessor

from profit.dataset.preprocessors.base_preprocessor import BasePreprocessor
from profit.dataset.preprocessors.egcn_preprocessor import EGCNPreprocessor
from profit.dataset.preprocessors.mol_preprocessor import MolPreprocessor
from profit.dataset.preprocessors.seq_preprocessor import SequencePreprocessor
from profit.dataset.preprocessors.transformer_preprocessor import TransformerPreprocessor

preprocess_method_dict = {
    'egcn': EGCNPreprocessor,
    'transformer': TransformerPreprocessor
}