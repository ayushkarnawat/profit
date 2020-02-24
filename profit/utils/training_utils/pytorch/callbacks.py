"""Pytorch callbacks. Modified from https://git.io/JvWqp.

Each callback implements an `on_train`, `on_epoch`, and `on_batch`
begin and end to determine the functionality of each callback when
those "events" occur.
"""

import warnings

from abc import ABC
from typing import Any, Dict, Optional

import numpy as np
import torch.nn as nn


class Callback(ABC):
    """Abstract base class for callbacks."""

    def __init__(self):
        self.model = None

    def set_model(self, model: nn.Module) -> None:
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
        """Called when the epoch ends."""

    def on_batch_begin(self, batch, logs=None):
        """Called when the batch starts.

        Params:
        -------
        batch: Tensor
            Current batch tensor.

        logs: dict or None, optional, default=None
            Key-value pairs of quantities to monitor.
        """
        pass

    def on_batch_end(self, batch, logs=None):
        """Called when the batch ends."""

    def on_train_begin(self, logs=None):
        """Called when training begins."""

    def on_train_end(self, logs=None):
        """Called when the training ends."""


class EarlyStopping(Callback):
    """Stop training when a monitored quantity has stopped improving.

    Params:
    -------
    monitor: str, default="val_loss"
        Quantity to be monitored.

    min_delta: float, default=0.0
        Minimum change in the monitored quantity to qualify as an
        improvement, i.e. an absolute change of less than `min_delta`,
        will count as no improvement.

    patience: int, default=0
        Number of epochs with no improvement after which training will
        be stopped.

    verbose: int, default=0
        Verbosity mode, 0 or 1.

    mode: str, default="auto"
        One of {auto, min, max}. In `min` mode, training will stop when
        the quantity monitored has stopped decreasing; in `max` mode it
        will stop when the quantity monitored has stopped increasing; in
        `auto` mode, the direction is automatically inferred from the
        name of the monitored quantity.

    strict: bool, default=True
        Whether to crash the training if `monitor` is not found in the
        metrics.
    """

    def __init__(self, monitor='val_loss', min_delta=0.0, patience=0,
                 verbose=0, mode='auto', strict=True):
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

    def check_metrics(self, logs: Dict[str, Any]) -> bool:
        """Check whether the quantity is being monitored in the logs.

        Params:
        -------
        logs: dict
            Key-value pairs of quantities to monitor.

        Returns:
        --------
        is_monitored: bool
            Whether or not the quantity is being monitored.
        """
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
        # Even if the quantity to be monitored is not within the logs, continue
        # to train w/o early stopping (if we are not strict about it).
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


class ModelCheckpoint(Callback):
    """Saves the model after each epoch.

    Params:
    -------
    filepath: str
        Path to save the model parameter weights file. Can contain named
        formatting options to be auto-filled.

    monitor: str, default="val_loss"
        Quantity to be monitored.

    verbose: int, default=0
        Verbosity mode, 0 or 1.

    save_top_k: int, default=1
        If `save_top_k=k`, the best `k` models according to the quantity
        monitored are saved. If `save_top_k=0`, no models will be saved.
        If `save_top_k=-1`, all models are saved. If `save_top_k >= 2`
        and the callback is called multiple times inside an epoch, the
        name of the saved file will be appended with a version count
        starting with `v0`. Note that the quantities monitored are
        checked every `period` epochs.

    save_weights_only: bool, default=False
        If True, then only the model's weights will be saved, else the
        full model is saved.

    mode: str, default="auto"
        One of {auto, min, max}. In `min` mode, training will stop when
        the quantity monitored has stopped decreasing; in `max` mode it
        will stop when the quantity monitored has stopped increasing; in
        `auto` mode, the direction is automatically inferred from the
        name of the monitored quantity.

    period: int, default=1
        Interval (number of epochs) between checkpoints.

    prefix: str, default=""
        Filepath prefix.
    """

    def __init__(self):
        pass


if __name__ == "__main__":
    clbk = EarlyStopping("val_loss", min_delta=0.3, patience=2, verbose=True)
    losses = [10, 9, 8, 8, 6, 4.3, 5, 4.4, 2.8, 2.5]
    for i, loss in enumerate(losses):
        stop = clbk.on_epoch_end(i, logs={"val_loss": loss})
        if stop:
            break