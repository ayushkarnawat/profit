import os

import h5py
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from typing import Any, Dict, Union


def _bytes_feature(value: Union[str, bytes]):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value: float):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value: Union[bool, int]):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize(example: Dict[str, Dict[str, Any]]):
    """Serialize an example within the dataset.
    
    Params:
    -------
    example: dict
        Dictionary with fields "array_name" that contains info about 
        the "data" and its "_type".

    Returns:
    --------
    out: str
        The serialized example (in bytes).
    """
    dset_item = {}
    for feature in example.keys():
        dset_item[feature] = example[feature]["_type"](example[feature]["data"])
        example_proto = tf.train.Example(features=tf.train.Features(feature=dset_item))
    return example_proto.SerializeToString()


def deserialize(example: Any):
    raise NotImplementedError


def save_to_tfrecords(dataset_fp):
    """Save a .h5 file to .tfrecords."""
    filename, _ = os.path.splitext(dataset_fp)
    with h5py.File(dataset_fp, "r") as h5file:
        n_examples = h5file[list(h5file.keys())[0]].shape[0]
        with tf.io.TFRecordWriter("{}.tfrecord".format(filename)) as writer:
            for row in tqdm(range(n_examples)):
                example = {arr_name: {"data": np.string_(h5file[arr_name][row]).tobytes(), 
                                      "_type": _bytes_feature} 
                           for arr_name in list(h5file.keys())}
                # Serialize + write the defined example into the dataset
                writer.write(serialize(example))