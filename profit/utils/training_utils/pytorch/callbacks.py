"""Pytorch callbacks. Modified from https://git.io/JvWqp.

Each callback implements an `on_train`, `on_epoch`, and `on_batch`
begin and end to determine the functionality of each callback when
those "events" occur.
"""

import os
import shutil
import warnings

from abc import ABC
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


class Callback(ABC):
    """Abstract base class for callbacks."""

    def __init__(self):
        self.model = None

    def set_model(self, model: nn.Module) -> None:
        """Sets the torch model."""
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
                warnings.warn(f"EarlyStopping mode {mode} is unknown, fallback "
                              "to auto mode.")
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
    savedir: str
        Directory to save the model parameter weights file. Can contain
        named formatting options to be auto-filled.

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

    def __init__(self, savedir: str, monitor="val_loss", verbose: int=0,
                 save_top_k: int=1, save_weights_only: bool=False,
                 mode: str="auto", period: int=1, prefix: str=""):
        super(ModelCheckpoint, self).__init__()

        if save_top_k and os.path.isdir(savedir) and len(os.listdir(savedir)) > 0:
            warnings.warn(f"Checkpoint directory {savedir} exists and is not "
                          "empty with save_top_k != 0. All files in this "
                          "directory will be deleted when a checkpoint is saved!")

        self.savedir = savedir
        os.makedirs(savedir, exist_ok=True)
        self.monitor = monitor
        self.verbose = verbose
        self.save_top_k = save_top_k
        self.save_weights_only = save_weights_only
        self.period = period
        self.prefix = prefix
        self.epochs_since_last_check = 0
        self.best_k_models = {} # {filename: monitor}
        self.kth_best_model = ''
        self.best = 0

        if mode not in ['auto', 'min', 'max']:
            if self.verbose > 0:
                warnings.warn(f"ModelCheckpoint mode {mode} is unknown, "
                              "fallback to auto mode.", RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.kth_value = np.Inf
            self.mode = 'min'
        elif mode == 'max':
            self.monitor_op = np.greater
            self.kth_value = -np.Inf
            self.mode = 'max'
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.kth_value = -np.Inf
                self.mode = 'max'
            else:
                self.monitor_op = np.less
                self.kth_value = np.Inf
                self.mode = 'min'

    def _del_model(self, filepath: str) -> None:
        """Delete the specified model."""
        dirpath = os.path.dirname(filepath)
        os.makedirs(dirpath, exist_ok=True)

        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)

    def _save_model(self, filepath: str) -> None:
        """Save the specified model."""
        dirpath = os.path.dirname(filepath)
        os.makedirs(dirpath, exist_ok=True)

        # Save the entire model or its weights. NOTE: It might be useful to save
        # other info such as current epoch, loss, model's state, optimizer's
        # state, etc. TODO: How do we get the epoch, loss, and optimizer info?
        # See: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        if self.save_weights_only:
            torch.save(self.model.state_dict(), filepath)
        else:
            torch.save(model, filepath)

    def check_monitor_top_k(self, current: float) -> bool:
        """Check the quantity monitored for improvement."""
        less_than_k_models = len(self.best_k_models.keys()) < self.save_top_k
        if less_than_k_models:
            return True
        return self.monitor_op(current, self.best_k_models[self.kth_best_model])

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]]=None):
        """Called when the epoch ends."""
        logs = logs or {}
        self.epochs_since_last_check += 1

        # No model to be saved
        if self.save_top_k == 0:
            return
        if self.epochs_since_last_check >= self.period:
            self.epochs_since_last_check = 0 # reset
            filepath = os.path.join(f"{self.savedir}", f"{self.prefix}_epoch{epoch}.ckpt")
            version_cnt = 0
            while os.path.isfile(filepath):
                # if this epoch was called before, make versions
                filepath = os.path.join(f"{self.savedir}", 
                                        f"{self.prefix}_epoch{epoch}_v{version_cnt}.ckpt")
                version_cnt += 1

            if self.save_top_k != -1:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn(f"Can save best model only with {self.monitor} "
                                  "available, skipping.", RuntimeWarning)
                else:
                    if self.check_monitor_top_k(current):
                        # Remove kth model
                        if len(self.best_k_models.keys()) == self.save_top_k:
                            delpath = self.kth_best_model
                            self.best_k_models.pop(self.kth_best_model)
                            self._del_model(delpath)

                        self.best_k_models[filepath] = current
                        if len(self.best_k_models.keys()) == self.save_top_k:
                            # monitor dict has reached k elements
                            if self.mode == 'min':
                                self.kth_best_model = max(self.best_k_models, 
                                                          key=self.best_k_models.get)
                            else:
                                self.kth_best_model = min(self.best_k_models, 
                                                          key=self.best_k_models.get)
                            self.kth_value = self.best_k_models[self.kth_best_model]

                        if self.mode == 'min':
                            self.best = min(self.best_k_models.values())
                        else:
                            self.best = max(self.best_k_models.values())
                        if self.verbose > 0:
                            print(f"Epoch {epoch:05d}: {self.monitor} reached "
                                  f"{current:0.5f} (best {self.best:0.5f}), "
                                  f"saving model to {filepath} as top {self.save_top_k}")
                        self._save_model(filepath)
                    else:
                        if self.verbose > 0:
                            print(f"Epoch {epoch:05d}: {self.monitor} was not "
                                  f"in the top {self.save_top_k}")
            else:
                if self.verbose > 0:
                    print(f"Epoch {epoch:05d}: saving model to {filepath}")
                self._save_model(filepath)


if __name__ == "__main__":
    # Create dummy model, no training occurs
    from profit.models.pytorch.egcn import EmbeddedGCN
    model = EmbeddedGCN(num_atoms=50, num_feats=63, num_outputs=1,
                        num_layers=2, units_conv=8, units_dense=8)

    # Setup callbacks
    stop_clbk = EarlyStopping("val_loss", min_delta=0.3, patience=2, verbose=1)
    save_clbk = ModelCheckpoint("ckpt_test/", monitor="val_loss", verbose=1, save_top_k=2)
    save_clbk.set_model(model)

    # Generate "fake" losses for the model
    losses = [10, 9, 8, 8, 6, 4.3, 5, 4.4, 2.8, 2.5]
    for i, loss in enumerate(losses):
        save_clbk.on_epoch_end(i, logs={"val_loss": loss})
        should_stop = stop_clbk.on_epoch_end(i, logs={"val_loss": loss})
        if should_stop:
            break