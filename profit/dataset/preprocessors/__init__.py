"""Feature generation and extraction (preprocessing)."""

from profit.dataset.preprocessors import base_preprocessor
from profit.dataset.preprocessors import egcn_preprocessor
from profit.dataset.preprocessors import lstm_preprocessor
from profit.dataset.preprocessors import mol_preprocessor
from profit.dataset.preprocessors import seq_preprocessor

from profit.dataset.preprocessors.base_preprocessor import BasePreprocessor
from profit.dataset.preprocessors.egcn_preprocessor import EGCNPreprocessor
from profit.dataset.preprocessors.lstm_preprocessor import LSTMPreprocessor
from profit.dataset.preprocessors.mol_preprocessor import MolPreprocessor
from profit.dataset.preprocessors.seq_preprocessor import SequencePreprocessor

preprocess_method_dict = {
    'egcn': EGCNPreprocessor,
    'lstm': LSTMPreprocessor
}
