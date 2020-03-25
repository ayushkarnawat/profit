"""Train GB1 protein engineering model."""

import torch
from torch.utils.data import DataLoader, Subset

from profit.dataset.splitters import split_method_dict
from profit.models.pytorch.egcn import EmbeddedGCN
from profit.utils.training_utils.pytorch.callbacks import EarlyStopping
from profit.utils.training_utils.pytorch.callbacks import ModelCheckpoint
from profit.utils.training_utils.pytorch.optimizers import AdamW

from data import load_dataset


# Preprocess + load the dataset
data = load_dataset('egcn', 'tertiary', labels='Fitness', num_data=5,
                    filetype='mdb', as_numpy=False)

# Determine which device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Stratify the dataset into train/val sets
# TODO: Use a stratified sampler to split the target labels equally into each
# subset. That is, both the train and validation datasets will have the same
# ratio of low/mid/high fitness variants as the full dataset in each batch.
# See: https://discuss.pytorch.org/t/29907/2
_dataset = torch.Tensor(len(data), 1) # hack to allow splitting to work properly
_labels = data[:]['arr_3'].view(-1).tolist()
# Create subset indicies
train_idx, val_idx = split_method_dict['stratified']().train_valid_split(
    _dataset, labels=_labels, frac_train=0.8, frac_val=0.2, return_idxs=True,
    task_type="auto", n_bins=10)
train_dataset = Subset(data, train_idx)
val_dataset = Subset(data, val_idx)

# For now, we hope that shuffling the dataset will be enough to make it random
# that some non-zero target labels are still introduced in each batch. Obviously
# this is not ideal, but, for the sake of quickness, works for now.
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

# Initialize model
num_atoms, num_feats = data[0]['arr_0'].shape
model = EmbeddedGCN(num_atoms, num_feats, num_outputs=1, num_layers=1,
                    units_conv=8, units_dense=8)

# Init callbacks
stop_clbk = EarlyStopping(patience=2, verbose=1)
save_clbk = ModelCheckpoint("bin/3gb1/egcn_fitness/", verbose=1,
                            save_weights_only=True, prefix="design0")
# Cumbersome, but required to ensure weights get saved properly.
# How do we ensure that the model (and its updated weights) are being used
# everytime we are sampling the new batch?
save_clbk.set_model(model)

# Construct loss function and optimizer
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = AdamW(model.parameters(), lr=1e-4)

print(f'Train on {len(train_idx)}, validate on {len(val_idx)}...')
for epoch in range(5):
    # Training loop
    model.train()
    train_loss_epoch = 0
    for batch_idx, batch in enumerate(train_loader):
        # Move feats/labels to gpu device (if available)
        atoms, adjms, dists, labels = [arr.to(device) for arr in batch.values()]
        # Forward pass though model
        train_y_pred = model([atoms, adjms, dists])

        # Compute and print loss
        loss = criterion(train_y_pred, labels)
        train_loss_epoch += loss.item()
        print(batch_idx, loss.item()) # loss per mini batch

        # Zero gradients, perform a backward pass, and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(epoch, train_loss_epoch)

    # Validation loop
    model.eval()
    val_loss_epoch = 0
    for j, val_batch in enumerate(val_loader):
        # Move feats/labels to gpu device (if available)
        val_atoms, val_adjms, val_dists, val_labels = [arr.to(device)\
            for arr in val_batch.values()]
        val_y_pred = model([val_atoms, val_adjms, val_dists])
        val_loss = criterion(val_y_pred, val_labels)
        val_loss_epoch += val_loss.item()
    print(epoch, val_loss_epoch)

    # Stop training (based off val loss) and save (top k) ckpts
    save_clbk.on_epoch_end(epoch, logs={"val_loss": val_loss_epoch})
    should_stop = stop_clbk.on_epoch_end(epoch, logs={"val_loss": val_loss_epoch})
    if should_stop:
        break

# See active sampling technique: https://github.com/rmunro/pytorch_active_learning
# Better, as it shows examples: https://github.com/ej0cl6/deep-active-learning/
