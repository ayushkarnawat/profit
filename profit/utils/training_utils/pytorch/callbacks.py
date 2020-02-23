"""Pytorch callbacks. Modified from: https://git.io/JvWq5

TODO: Implement as proper callbacks (similar to tf/keras), where after 
each `on_epoch_end` updates the logs of the quantity to be monitored. 
See https://git.io/JvWqp for more details on implementation.
"""

import warnings

from abc import ABC
from typing import Any, Dict, Optional

import numpy as np
import torch


class Callback(ABC):
    """Abstract base class used to build callbacks."""

    def __init__(self):
        self.validation_data = None
        self.model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]]=None):
        """Called when the epoch begins.

        Params:
        -------
        epoch: int
            Current epoch
        
        logs: dict or None, optional, default=None
            Key-value pairs of quantities to monitor.
        
        Example:
        --------
        ```python
        on_epoch_begin(epoch=2, logs={'val_loss': 0.2})
        ```
        """
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        """
        called when the batch starts.
        Args:
            batch (Tensor): current batch tensor
            logs (dict): key-value pairs of quantities to monitor
        """
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class EarlyStopping(Callback):
    """Stop training when a monitored quantity has stopped improving.

    Args:
        monitor (str): quantity to be monitored. Default: ``'val_loss'``.
        min_delta (float): minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than `min_delta`, will count as no
            improvement. Default: ``0``.
        patience (int): number of epochs with no improvement
            after which training will be stopped. Default: ``0``.
        verbose (bool): verbosity mode. Default: ``0``.
        mode (str): one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity. Default: ``'auto'``.
        strict (bool): whether to crash the training if `monitor` is
            not found in the metrics. Default: ``True``.
    """

    def __init__(self, monitor='val_loss',
                 min_delta=0.0, patience=0, verbose=0, mode='auto', strict=True):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.strict = strict
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0

        if mode not in ['auto', 'min', 'max']:
            if self.verbose > 0:
                print(f'EarlyStopping mode {mode} is unknown, fallback to auto mode.')
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

        self.on_train_begin()

    def check_metrics(self, logs):
        monitor_val = logs.get(self.monitor)
        error_msg = (f'Early stopping conditioned on metric `{self.monitor}`'
                     f' which is not available. Available metrics are:'
                     f' `{"`, `".join(list(logs.keys()))}`')

        if monitor_val is None:
            if self.strict:
                raise RuntimeError(error_msg)
            if self.verbose > 0:
                warnings.warn(error_msg, RuntimeWarning)
            return False
        return True

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        stop_training = False
        if not self.check_metrics(logs):
            return stop_training

        current = logs.get(self.monitor)
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                stop_training = True
                self.on_train_end()
        return stop_training

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f'Epoch {self.stopped_epoch + 1:05d}: early stopping')


# class EarlyStopping(object):
#     """Stops training when a monitored quantity has stoped improving.
    
#     Params:
#     -------
#     monitor: str, default="val_loss"
#         Quantity to be monitored. Ignored for now. TODO: Implement

#     min_delta: float, default=0
#         Minimum change in the monitored quantity to qualify as an 
#         improvement, i.e. an absolute change of less than `min_delta`, 
#         will count as no improvement.
    
#     patience: int, default=5
#         Number of epochs with no improvement after which training will 
#         be stopped.
    
#     verbose: bool, default=False
#         Verbosity mode. If True, prints info about each `monitor` 
#         quantity improvement.
#     """

#     def __init__(self, monitor: str="val_loss", min_delta: float=0., 
#                  patience: int=5, verbose: bool=False):
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.delta = min_delta

#     def on_epoch_end(self, val_loss, model):
#         """Called when the epoch ends.
        
#         Determines whether or not to save the model based on historical 
#         performance of the val_loss (aka whenever it decreases).
#         """
#         score = -val_loss

#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#             self.counter = 0

#     def save_checkpoint(self, val_loss, model):
#         """Saves model when validation loss decrease."""
#         if self.verbose:
#             print(f"Validation loss decreased ({self.val_loss_min:.6f} " + 
#                    "--> {val_loss:.6f}). Saving model ...")
#         torch.save(model.state_dict(), 'checkpoint.pt')
#         self.val_loss_min = val_loss