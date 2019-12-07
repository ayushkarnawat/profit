import numpy as np
from typing import List, Union


def broadcast_array(arr: List[Union[List[int], List[float], np.ndarray]], values=0) -> np.ndarray:
    """Cast all arrays to same output size.
    
    Params:
    -------
    arr: list of int or float or np.ndarray
        Arbitrarily sized list of lists.

    Returns:
    --------
    padded_arr: np.ndarray
        Padded array with final shape equal to the max value across 
        each dim in the original list. 
    """
    # Determine how much to pad each example
    shapes = np.array([np.array(ex).shape for ex in arr])
    out_size = np.max(shapes, axis=0) # max across all dimensions
    pad_widths = [list(out_size - shape) for shape in shapes] # i.e [(25,0), (0,0), (3,0)]

    # Pad (w/ zeros) to computed size (aka max size across each dimension)
    # NOTE: Formatting of pad_widths is necessary so that the padded values are 
    # added to the end of each dimension. The format is, as follows:
    # 
    # [[(0, pad_widths[0][0]), (0, pad_widths[0][1]), ..., (0, pad_widths[0][m])]
    #  [(0, pad_widths[1][0]), (0, pad_widths[1][1]), ..., (0, pad_widths[1][m])]
    #  [(0, pad_widths[2][0]), (0, pad_widths[2][1]), ..., (0, pad_widths[2][m])]
    #             .                       .           .              .           
    #             .                       .             .            .           
    #             .                       .               .          .           
    #  [(0, pad_widths[n][0]), (0, pad_widths[n][1]), ..., (0, pad_widths[n][m])]]
    # 
    # where each row is an example, and each tuple in a column signify how much 
    # each axis is going to be padded. Note that |N| is the total number of examples 
    # in the dataset and |M| is the total number of dimensions, aka len(out_size).
    pad_widths_fmt = [[(0,pad_size) for pad_size in pad_width] for pad_width in pad_widths]
    padded_arr = [np.pad(example_feat, pad_width, mode="constant", constant_values=values)
                  for pad_width, example_feat in zip(pad_widths_fmt, arr)]
    return np.array(padded_arr)