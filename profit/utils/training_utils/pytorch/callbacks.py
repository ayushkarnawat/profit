"""Pytorch callbacks. Modified from: https://git.io/JvWq5

TODO: Implement as proper callbacks (similar to tf/keras), where after 
each `on_epoch_end` updates the logs of the quantity to be monitored. 
See https://git.io/JvWqp for more details on implementation.
"""

import numpy as np
import torch


class EarlyStopping(object):
    """Stops training when a monitored quantity has stoped improving.
    
    Params:
    -------
    monitor: str, default="val_loss"
        Quantity to be monitored. Ignored for now. TODO: Implement

    min_delta: float, default=0
        Minimum change in the monitored quantity to qualify as an 
        improvement, i.e. an absolute change of less than `min_delta`, 
        will count as no improvement.
    
    patience: int, default=5
        Number of epochs with no improvement after which training will 
        be stopped.
    
    verbose: bool, default=False
        Verbosity mode. If True, prints info about each `monitor` 
        quantity improvement.
    """

    def __init__(self, monitor: str="val_loss", min_delta: float=0., 
                 patience: int=5, verbose: bool=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = min_delta

    def on_epoch_end(self, val_loss, model):
        """Called when the epoch ends.
        
        Determines whether or not to save the model based on historical 
        performance of the val_loss (aka whenever it decreases).
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} " + 
                   "--> {val_loss:.6f}). Saving model ...")
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss