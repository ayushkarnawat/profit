from profit.utils.data_utils import cacher
from profit.utils.data_utils import cast
from profit.utils.data_utils import datasets
from profit.utils.data_utils import example_pb2
from profit.utils.data_utils import peptides
from profit.utils.data_utils import serializers
from profit.utils.data_utils import substitution_matrices
from profit.utils.data_utils import tfreader
from profit.utils.data_utils import tokenizers
from profit.utils.data_utils import vocabs

from profit.utils.data_utils.cacher import CacheNamePolicy

from profit.utils.data_utils.cast import broadcast_array

from profit.utils.data_utils.datasets import TensorflowHDF5Dataset
from profit.utils.data_utils.datasets import TensorflowLMDBDataset
from profit.utils.data_utils.datasets import TensorflowNumpyDataset
from profit.utils.data_utils.datasets import TFRecordsDataset
from profit.utils.data_utils.datasets import TorchHDF5Dataset
from profit.utils.data_utils.datasets import TorchLMDBDataset
from profit.utils.data_utils.datasets import TorchNumpyDataset
from profit.utils.data_utils.datasets import TorchTFRecordsDataset

from profit.utils.data_utils.peptides import *

from profit.utils.data_utils.serializers import HDF5Serializer
from profit.utils.data_utils.serializers import LMDBSerializer
from profit.utils.data_utils.serializers import NumpySerializer
from profit.utils.data_utils.serializers import TFRecordsSerializer

from profit.utils.data_utils.substitution_matrices import BLOSUM62

from profit.utils.data_utils.tfreader import tfrecord_iterator
from profit.utils.data_utils.tfreader import tfrecord_loader

from profit.utils.data_utils.tokenizers import BaseTokenizer
from profit.utils.data_utils.tokenizers import AminoAcidTokenizer

from profit.utils.data_utils.vocabs import VOCABS

serialize_method_dict = {
    "h5": HDF5Serializer,
    "hdf5": HDF5Serializer,
    "lmdb": LMDBSerializer,
    "mdb": LMDBSerializer,
    "npz": NumpySerializer,
    "tfrecords": TFRecordsSerializer
}
