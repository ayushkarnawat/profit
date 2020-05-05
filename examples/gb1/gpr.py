"""Generic gaussian process regressor (GPR) on 3GB1 fitness dataset."""

import os
import time
import pickle as pkl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Subset

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from profit.dataset.splitters import split_method_dict
from profit.models.torch.gpr import SequenceGPR
from profit.utils.data_utils.tokenizers import AminoAcidTokenizer

from data import load_dataset


timestep = time.strftime("%Y-%b-%d-%H:%M:%S", time.gmtime())
use_substitution = True     # use aa substitution kernel (does not learn params)
save_model = False          # save gp model
savedir = "bin/3gb1/gpr"    # dir for saving model
plot_seq = False            # plot amino acid seq as xticks

# Preprocess + load the dataset
dataset = load_dataset("lstm", "primary", labels="Fitness", num_data=-1,
                       filetype="mdb", as_numpy=False, vocab="aa20")
_dataset = dataset[:]["arr_0"]
_labels = dataset[:]["arr_1"].view(-1)
# Remove samples below a certain threshold
high_idx = torch.where(_labels > _labels.mean())
dataset = Subset(dataset, sorted(high_idx))
_dataset = _dataset[high_idx]
_labels = _labels[high_idx]

# Shuffle, split, and batch
splits = {"train": 1.0, "valid": 0.0}
subset_idx = split_method_dict["stratified"]().train_valid_split(
    _dataset, _labels.tolist(), frac_train=splits.get("train", 1.0),
    frac_valid=splits.get("valid", 0.0), n_bins=10, return_idxs=True)
stratified = {split: Subset(dataset, sorted(idx))
              for split, idx in zip(splits.keys(), subset_idx)}
train_X, train_y = stratified["train"][:].values()

# Instantiate a Gaussian Process model
if use_substitution:
    gp = SequenceGPR()
else:
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit to data; optimize kernel params using Maximum Likelihood Estimation (MLE)
gp.fit(train_X, train_y)

# Save GPR
if save_model:
    os.makedirs(savedir, exist_ok=True)
    filepath = os.path.join(savedir, f"{timestep}.pt")
    _ = gp.save(filepath) if use_substitution else pkl.dumps(gp, filepath)

# Make prediction (mu) on whole sample space (ask for std as well)
y_pred, sigma = gp.predict(_dataset, return_std=True)
if isinstance(y_pred, torch.Tensor):
    y_pred = y_pred.numpy()
if isinstance(sigma, torch.Tensor):
    sigma = sigma.numpy()

tokenizer = AminoAcidTokenizer("aa20")
pos = [38, 39, 40, 53]
df = pd.DataFrame({
    "is_train": [1 if idx in subset_idx[0] else 0 for idx in range(len(dataset))],
    "seq": ["".join(tokenizer.decode(seq)) for seq in _dataset[:, pos].numpy()],
    "true": _labels.numpy(),
    "pred": y_pred.flatten(),
    "sigma": sigma.flatten(),
})

# If x-axis labels are seq, sort df by seq (in alphabetical order) for "better"
# visualization; if plotting via index, no need for resorting.
if plot_seq:
    df = df.sort_values("seq", ascending=True)
train_only = df.loc[df["is_train"] == 1]
valid_only = df.loc[df["is_train"] == 0]

# Determine how well the regressor fit to the dataset
train_mse = np.mean(np.square((train_only["pred"] - train_only["true"])))
valid_mse = np.mean(np.square((valid_only["pred"] - valid_only["true"])))
print(f"Train MSE: {train_mse}\t Valid MSE: {valid_mse}")

# Plot observations, prediction and 95% confidence interval (2\sigma).
# NOTE: We plot the whole sequence to avoid erratic line jumps
plt.figure()
if plot_seq:
    # If using mutated chars amino acid seq as the xtick labels
    plt.plot(df["seq"].values, df["pred"].values, "b-", label="Prediction")
    plt.plot(df["seq"].values, df["true"].values, "r:", label="True")
    plt.plot(train_only["seq"].values, train_only["true"].values, "r.",
             markersize=10, label="Observations")
    plt.fill(np.concatenate([df["seq"].values, df["seq"].values[::-1]]),
             np.concatenate([df["pred"].values - 1.9600 * df["sigma"].values,
                             (df["pred"].values + 1.9600 * df["sigma"].values)[::-1]]),
             alpha=.5, fc="b", ec="None", label="95% confidence interval")
    plt.xticks(rotation=90)
else:
    # If using index as the x-axis
    plt.plot(df.index, df["pred"].values, "b-", label="Prediction")
    plt.plot(df.index, df["true"].values, "r:", label="True")
    plt.plot(train_only.index, train_only["true"].values, "r.",
             markersize=10, label="Observations")
    plt.fill(np.concatenate([df.index, df.index[::-1]]),
             np.concatenate([df["pred"].values - 1.9600 * df["sigma"].values,
                             (df["pred"].values + 1.9600 * df["sigma"].values)[::-1]]),
             alpha=.5, fc="b", ec="None", label="95% confidence interval")
plt.xlabel("Sequence ($x$)")
plt.ylabel("Fitness ($y$)")
plt.title("Predicting protein fitness using GPR (PDB: 3GB1)")
plt.legend(loc="upper left")
plt.show()
