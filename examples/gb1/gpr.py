"""Generic gaussian process regressor (GPR) on 3GB1 fitness dataset."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from profit.dataset.splitters import split_method_dict
from profit.utils.data_utils.tokenizers import AminoAcidTokenizer

from data import load_dataset
from seq_gp import SequenceGPR


# Preprocess + load the dataset
dataset = load_dataset("lstm", "primary", labels="Fitness", num_data=-1,
                       filetype="mdb", as_numpy=True, vocab="aa20")
# dataset = [arr[-10:] for arr in dataset]

# Shuffle, split, and batch
splits = ["train", "valid"]
subset_idx = split_method_dict["stratified"]().train_valid_split(dataset[0], \
    dataset[-1].flatten(), frac_train=0.8, frac_val=0.2, n_bins=10, return_idxs=True)
stratified = {split: [arr[idx] for arr in dataset]
              for split, idx in zip(splits, subset_idx)}

train_X, train_y = stratified["train"]
val_X, val_y = stratified["valid"]

# Instantiate a Gaussian Process model
use_substitution = True
if use_substitution:
    gp = SequenceGPR()
else:
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit to data and optimize kernel params using Maximum Likelihood Estimation (MLE)
gp.fit(train_X, train_y)

# Make prediction (mu) on whole sample space (ask for std as well)
y_pred, sigma = gp.predict(dataset[0], return_std=True)

tokenizer = AminoAcidTokenizer("aa20")
seqs_4char = []
for encoded_seq in dataset[0]:
    seq = tokenizer.decode(encoded_seq)
    seqs_4char.append(seq[38] + seq[39] + seq[40] + seq[53])
df = pd.DataFrame(columns=["seq", "true", "pred", "sigma"])
df["seq"] = seqs_4char
df["true"] = dataset[-1]
df["pred"] = y_pred
df["sigma"] = sigma
df["is_train"] = [1 if idx in subset_idx[0] else 0 for idx in range(dataset[0].shape[0])]

# If x-axis is seq, sort df by seq (in alphabetical order) for "better"
# visualization. If plotting via index, no need for resorting.
plot_seq = False
if plot_seq:
    df = df.sort_values("seq", ascending=True)
train_only = df.loc[df["is_train"] == 1]
val_only = df.loc[df["is_train"] == 0]

# Determine how well the regressor fit to the dataset
mse = np.mean(np.square((val_only["pred"] - val_only["true"])))
print(f"MSE: {mse}")

# Plot observations, prediction and 95% confidence interval (2\sigma).
# NOTE: We plot the whole sequence to avoid erratic line jumps
plt.figure()
if plot_seq:
    # If using 4char amino acid seq as the x-axis values
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
