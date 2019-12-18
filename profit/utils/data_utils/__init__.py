from profit.utils.data_utils import cacher
from profit.utils.data_utils import cast
from profit.utils.data_utils import serializers
from profit.utils.data_utils import vocabs

from profit.utils.data_utils.cacher import CacheNamePolicy
from profit.utils.data_utils.cast import broadcast_array

from profit.utils.data_utils.serializers import HDF5Serializer
from profit.utils.data_utils.serializers import LMDBSerializer
from profit.utils.data_utils.serializers import NumpySerializer
from profit.utils.data_utils.serializers import TFRecordsSerializer

from profit.utils.data_utils.vocabs import AA1_VOCAB
from profit.utils.data_utils.vocabs import AA3_VOCAB
from profit.utils.data_utils.vocabs import FLIPPED_AA1
from profit.utils.data_utils.vocabs import FLIPPED_AA3

serialize_method_dict = {
    "h5": HDF5Serializer,
    "hdf5": HDF5Serializer,
    "lmdb": LMDBSerializer,
    "mdb": LMDBSerializer,
    "npz": NumpySerializer,
    "tfrecords": TFRecordsSerializer
}