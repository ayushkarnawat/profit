import io
import os
import struct
import warnings

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from profit.utils.data_utils import example_pb2


def tfrecord_iterator(data_path: str, index_path: Optional[str] = None,
                      shard: Optional[Tuple[int, int]] = None) -> Iterable[memoryview]:
    """Create an iterator over the tfrecord dataset.

    Since the tfrecords file stores each example as bytes, we can
    define an iterator over `datum_bytes_view`, which is a memoryview
    object referencing the bytes.

    Params:
    -------
    data_path: str
        TFRecord file path.

    index_path: str, optional, default=None
        Index file path. Can be set to None if no file is available.

    shard: tuple of ints, optional, default=None
        A tuple (index, count) representing worker_id and num_workers
        count. Necessary to evenly split/shard the dataset among many
        workers (i.e. >1).

    Yields:
    -------
    datum_bytes_view: memoryview
        Object referencing the specified `datum_bytes` contained in the
        file (for a single record).
    """
    file = io.open(data_path, "rb")

    length_bytes = bytearray(8)
    crc_bytes = bytearray(4)
    datum_bytes = bytearray(1024*1024)

    def read_records(start_offset=None, end_offset=None):
        nonlocal length_bytes, crc_bytes, datum_bytes
        if start_offset is not None:
            file.seek(start_offset)
        if end_offset is None:
            end_offset = os.path.getsize(data_path)
        while file.tell() < end_offset:
            if file.readinto(length_bytes) != 8:
                raise RuntimeError("Failed to read the record size.")
            if file.readinto(crc_bytes) != 4:
                raise RuntimeError("Failed to read the start token.")
            length, = struct.unpack("<Q", length_bytes)
            # Grow datum_bytes (at 1.5x) if default size is not enough to read record
            if length > len(datum_bytes):
                datum_bytes = datum_bytes.zfill(int(length * 1.5))
            datum_bytes_view = memoryview(datum_bytes)[:length]
            if file.readinto(datum_bytes_view) != length:
                raise RuntimeError("Failed to read the record.")
            if file.readinto(crc_bytes) != 4:
                raise RuntimeError("Failed to read the end token.")
            yield datum_bytes_view

    if index_path is None:
        warnings.warn(f"Could not find index associated with '{data_path}' - " +
                      "data sharding will be unavailable! \033[93mNOTE\033[0m: " +
                      "Using num_workers > 1 in `torch.utils.data.DataLoader()` " +
                      "might yield duplicate data!")
        yield from read_records()
    else:
        index = np.loadtxt(index_path, dtype=np.int64)[:, 0]
        if shard is None:
            offset = np.random.choice(index)
            yield from read_records(offset)
            yield from read_records(0, offset)
        else:
            num_records = len(index)
            shard_idx, shard_count = shard
            start_index = (num_records * shard_idx) // shard_count
            end_index = (num_records * (shard_idx + 1)) // shard_count
            start_byte = index[start_index]
            end_byte = index[end_index] if end_index < num_records else None
            yield from read_records(start_byte, end_byte)
    file.close()


def tfrecord_loader(data_path: str, index_path: Optional[str] = None,
                    shard: Optional[Tuple[int, int]] = None) \
                    -> Iterable[Dict[str, np.ndarray]]:
    """Create an iterator over the (decoded) examples contained within
    the dataset.

    Decodes the raw bytes of the features (contained within the
    dataset) into its respective format.

    Params:
    -------
    data_path: str
        TFRecord file path.

    index_path: str, optional, default=None
        Index file path. Can be set to None if no file is available.

    shard: tuple of ints, optional, default=None
        A tuple (index, count) representing worker_id and num_workers
        count. Necessary to evenly split/shard the dataset among many
        workers (i.e. >1).

    Yields:
    -------
    features: dict of {str, np.ndarray}
        Decoded bytes of the features into its respective data type
        (for an individual record).
    """
    record_iterator = tfrecord_iterator(data_path, index_path, shard)

    for record in record_iterator:
        example = example_pb2.Example()
        example.ParseFromString(record)

        features = {}
        for key in list(example.features.feature.keys()):
            # NOTE: We assume that each key in the example has only one field
            # (either "bytes_list", "float_list", or "int64_list")!
            field = example.features.feature[key].ListFields()[0]
            tf_typename, value = field[0].name, field[1].value

            # Decode raw bytes into respective data types
            if tf_typename == "bytes_list":
                value = np.frombuffer(value[0])
            elif tf_typename == "float_list":
                value = np.array(value, dtype=np.float32)
            elif tf_typename == "int64_list":
                value = np.array(value, dtype=np.int32)
            features[key] = value
        yield features
