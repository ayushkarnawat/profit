import os
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from typing import Any, Dict, List, Union


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


def dataset_to_tfrecords(data: List[np.ndarray], save_filepath: str):
    """Save a list of np.ndarray (aka dataset) into a .tfrecords file.

    NOTE: TFRecords flatten each ndarray before saving them as bytes 
    feature. To combat this, maybe we should save the shape dims as 
    well, and reshape them back after reading from the tfrecords.
    
    Params:
    -------
    data: list of np.ndarray
        Ndarray's to save. The first channel of each ndarray should  
        contain the number of examples in the dataset. 

    save_filepath: str
        Save filename path. Should end in '.tfrecords'.
    """
    assert save_filepath.endswith(".tfrecords")
    n_examples = data[0].shape[0]
    with tf.io.TFRecordWriter(save_filepath) as writer:
        for row in tqdm(range(n_examples), total=n_examples):
            example = {"arr_{}".format(idx): {"data": np.string_(nparr[row]).tobytes(), 
                                              "_type": _bytes_feature} 
                       for idx, nparr in enumerate(data)}
            # Write serialized example into the dataset
            writer.write(serialize(example))