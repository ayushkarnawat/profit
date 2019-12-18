import numpy as np
import tensorflow as tf

from data import load_dataset
from profit.models.gcn import GCN
from profit.dataset.splitters.stratified_splitter import StratifiedSplitter


# Preprocess + load the dataset
data = load_dataset('gcn', 'tertiary', labels='Fitness', num_data=10, \
    filetype='tfrecords', as_numpy=True)

# Shuffle, split and batch
train_idx, val_idx = StratifiedSplitter().train_valid_split(data[0], \
    labels=data[-1].flatten(), return_idxs=True)
train_data = []
val_data = []
for arr in data:
    train_data.append(arr[train_idx])
    val_data.append(arr[val_idx])

train_X = train_data[:-1]
train_y = train_data[-1]
val_X = val_data[:-1]
val_y = val_data[-1]

# Initialize GCN model (really hacky), it also assumes we have the data loaded 
# in memory, which is the wrong approach. Instead, we should peek into the 
# shape defined in the TF tensors.
num_atoms, num_feats = train_data[0].shape[1], train_data[0].shape[2]
labels = train_data[-1]
num_outputs = labels.shape[1]
labels_std = np.std(labels, axis=0)
model = GCN(num_atoms, num_feats, num_outputs=num_outputs, std=labels_std).get_model()

# Fit model and report metrics
model.fit(train_X, train_y, batch_size=5, epochs=5, shuffle=True, 
          validation_data=(val_X, val_y), verbose=2)