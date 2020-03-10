"""Train 3gb1 LSTM protein engineering model."""

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from profit.dataset.splitters import split_method_dict
from profit.models.pytorch.lstm import LSTMModel
from profit.utils.data_utils.tokenizers import AminoAcidTokenizer
from profit.utils.training_utils.pytorch.optimizers import AdamW
from profit.utils.training_utils.pytorch.callbacks import EarlyStopping
from profit.utils.training_utils.pytorch.callbacks import ModelCheckpoint

from data import load_dataset


# Determine which device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocess + load the dataset
data = load_dataset('lstm', 'primary', labels='Fitness', num_data=-1,
                    filetype='mdb', as_numpy=False)

# Stratify the dataset into train/val sets
# NOTE: We use a stratified sampler to split the target labels equally into each
# subset. That is, both the train and validation datasets will have the same
# ratio of low/mid/high fitness variants as the full dataset in each batch.
# See: https://discuss.pytorch.org/t/29907/2
_dataset = data[:]["arr_0"]
_labels = data[:]['arr_1'].view(-1)
# Create subset indicies
train_idx, val_idx = split_method_dict['stratified']().train_valid_split(
    _dataset, labels=_labels.tolist(), frac_train=0.8, frac_val=0.2,
    return_idxs=True, n_bins=10)
train_dataset = Subset(data, sorted(train_idx))
val_dataset = Subset(data, sorted(val_idx))

# Compute sample weight (each sample should get its own weight)
def stratified_sampler(labels: torch.Tensor, nbins: int = 10) -> WeightedRandomSampler:
    bin_labels = torch.tensor(pd.qcut(labels.tolist(), q=nbins,
                                      labels=False, duplicates='drop'))
    class_sample_count = torch.tensor(
        [(bin_labels == t).sum() for t in torch.unique(bin_labels, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.zeros_like(labels)
    for t in torch.unique(bin_labels):
        samples_weight[bin_labels == t] = weight[t]
    return WeightedRandomSampler(samples_weight, len(samples_weight))

# Create sampler and loader
train_sampler = stratified_sampler(train_dataset[:]['arr_1'].view(-1))
train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=128)

# Initialize model
vocab_size = AminoAcidTokenizer("iupac1").vocab_size
model = LSTMModel(vocab_size, input_size=64, hidden_size=256, num_layers=3,
                  hidden_dropout=0.1)

# Init callbacks
stop_clbk = EarlyStopping(patience=3, verbose=1)
save_clbk = ModelCheckpoint("results/3gb1/lstm_fitness/", verbose=1,
                            save_weights_only=True, prefix="design0")
# Cumbersome, but required to ensure weights get saved properly.
# How do we ensure that the model (and its updated weights) are being used
# everytime we are sampling the new batch?
save_clbk.set_model(model)

# Construct loss function and optimizer
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = AdamW(model.parameters(), lr=1e-3)

print(f'Train on {len(train_idx)}, validate on {len(val_idx)}...')
# PSEUDOCODE: Until the convergeg criteria is not met: i.e. acqusition func. didn't change
# PSEUDOCODE: Update the prefix in the model saving such that it is design{idx}
for epoch in range(50):
    # Training loop
    model.train()
    train_loss_epoch = 0
    for batch_idx, batch in enumerate(train_loader):
        # Move feats/labels to gpu device (if available)
        feats, labels = [arr.to(device) for arr in batch.values()]
        # Forward pass though model
        train_y_pred = model(feats)

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
        val_feats, val_labels = [arr.to(device) for arr in val_batch.values()]
        val_y_pred = model(val_feats)
        val_loss = criterion(val_y_pred, val_labels)
        val_loss_epoch += val_loss.item()
    print(epoch, val_loss_epoch)

    # Stop training (based off val loss) and save (top k) ckpts
    save_clbk.on_epoch_end(epoch, logs={"val_loss": val_loss_epoch})
    should_stop = stop_clbk.on_epoch_end(epoch, logs={"val_loss": val_loss_epoch})
    if should_stop:
        break
