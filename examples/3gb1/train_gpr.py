import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from data import load_dataset
from profit.dataset.splitters import split_method_dict
from profit.utils.data_utils.tokenizers import AminoAcidTokenizer


# Preprocess + load the dataset
dataset = load_dataset('transformer', 'primary', labels='Fitness', num_data=-1, \
    filetype='tfrecords', as_numpy=True)

# Shuffle, split, and batch
train_idx, val_idx = split_method_dict['random']().train_valid_split(dataset[0], \
    labels=dataset[-1].flatten(), return_idxs=True)
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

# Make the prediction (ask for MSE as well)
y_pred, sigma = gp.predict(val_X, return_std=True)

fitness_max_idx = np.argmax(y_pred)
print(fitness_max_idx, val_X[fitness_max_idx])
# tokenizer = AminoAcidTokenizer('iupac1')
# seq = "".join(tokenizer.decode(val_X[fitness_max_idx]))
# print(np.max(y_pred), seq)
# print(seq[38] + seq[39] + seq[40] + seq[53])
# rmse = np.sqrt(np.mean(np.square((y_pred - val_y)))) * 1.0
# print(rmse)