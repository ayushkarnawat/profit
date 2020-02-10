"""
Modified from: https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-noisy-targets-py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from data import load_dataset
from profit.dataset.splitters import split_method_dict
from profit.utils.data_utils.tokenizers import AminoAcidTokenizer


# Preprocess + load the dataset
dataset = load_dataset('transformer', 'primary', labels='Fitness', num_data=-1, \
    filetype='tfrecords', as_numpy=True)
# dataset = [arr[-50:] for arr in dataset]

# Shuffle, split, and batch
train_idx, val_idx = split_method_dict['random']().train_valid_split(dataset[0], \
    labels=dataset[-1].flatten(), frac_train=0.8, frac_val=0.2, return_idxs=True)
train_data = []
val_data = []
for arr in dataset:
    train_data.append(arr[train_idx])
    val_data.append(arr[val_idx])

train_X, train_y = train_data[0], train_data[1]
val_X, val_y = val_data[0], val_data[1]

# Instantiate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(train_X, train_y)

# Make prediction on whole sample space (ask for MSE as well)
y_pred, sigma = gp.predict(dataset[0], return_std=True)

seqs_4char = []
tokenizer = AminoAcidTokenizer('iupac1')
for encoded_seq in dataset[0]:
    seq = tokenizer.decode(encoded_seq)
    seqs_4char.append(seq[38] + seq[39] + seq[40] + seq[53])
df = pd.DataFrame(columns=['seq', 'true', 'pred', 'sigma'])
df['seq'] = seqs_4char
df['true'] = dataset[-1]
df['pred'] = y_pred
df['sigma'] = sigma
df['is_train'] = [1 if idx in train_idx else 0 for idx in range(dataset[0].shape[0])]

# If x-axis is seq, sort df by seq (in alphabetical order) for "better" visualization
# If plotting via index, no need for resorting.
use_seq = True
if use_seq:
    df = df.sort_values('seq', ascending=True)
train_only = df.loc[df['is_train'] == 1]
val_only = df.loc[df['is_train'] == 0]

# Determine how well the regressor fit to the dataset
rmse = np.sqrt(np.mean(np.square((val_only['pred'] - val_only['true'])))) * 1.0
print(f"RMSE: {rmse}")

# Plot the observations, the prediction and the 95% confidence interval based on
# the MSE. Observe how "well" the model trained
plt.figure()
if use_seq:
    # If using 4char amino acid seq as the x-axis values
    plt.plot(df['seq'].values, df['pred'].values, 'b-', label='Prediction') # Plot whole seq space to avoid erratic line jumps
    plt.plot(df['seq'].values, df['true'].values, 'r:', label='True')
    plt.plot(train_only['seq'].values, train_only['true'].values, 'r.', markersize=10, label='Observations')
    plt.fill(np.concatenate([df['seq'].values, df['seq'].values[::-1]]),
             np.concatenate([df['pred'].values - 1.9600 * df['sigma'].values,
                            (df['pred'].values + 1.9600 * df['sigma'].values)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xticks(rotation=90)
else:
    # If using index as the x-axis
    plt.plot(df.index, df['pred'].values, 'b-', label='Prediction') # Plot whole seq space to avoid erratic line jumps
    plt.plot(df.index, df['true'].values, 'r:', label='True')
    plt.plot(train_only.index, train_only['true'].values, 'r.', markersize=10, label='Observations')
    plt.fill(np.concatenate([df.index, df.index[::-1]]),
             np.concatenate([df['pred'].values - 1.9600 * df['sigma'].values,
                            (df['pred'].values + 1.9600 * df['sigma'].values)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('Sequence ($x$)')
plt.ylabel('Fitness ($y$)')
plt.title('Predicting protein fitness using GPR (PDB: 3GB1)')
plt.legend(loc='upper left')
plt.show()