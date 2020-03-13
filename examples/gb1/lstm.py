"""Train GB1 LSTM protein engineering model.

References:
[1] https://git.io/Jv6kw
[2] https://git.io/Jv6ko
"""

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocess + load the dataset
data = load_dataset("lstm", "primary", labels="Fitness", num_data=-1,
                    filetype="mdb", as_numpy=False)

# Stratify the dataset into train/val sets
# NOTE: We use a stratified sampler to split the target labels equally into each
# subset. That is, both the train and validation datasets will have the same
# ratio of low/mid/high fitness variants as the full dataset in each batch.
# See: https://discuss.pytorch.org/t/29907/2
_dataset = data[:]["arr_0"]
_labels = data[:]["arr_1"].view(-1)
# Create subset indicies
train_idx, val_idx = split_method_dict["stratified"]().train_valid_split(
    _dataset, labels=_labels.tolist(), frac_train=0.8, frac_val=0.2,
    return_idxs=True, n_bins=10)
train_dataset = Subset(data, sorted(train_idx))
val_dataset = Subset(data, sorted(val_idx))

# Compute sample weight (each sample should get its own weight)
def stratified_sampler(labels: torch.Tensor, nbins: int = 10) -> WeightedRandomSampler:
    bin_labels = torch.tensor(pd.qcut(labels.tolist(), q=nbins,
                                      labels=False, duplicates="drop"))
    class_sample_count = torch.tensor(
        [(bin_labels == t).sum() for t in torch.unique(bin_labels, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.zeros_like(labels)
    for t in torch.unique(bin_labels):
        samples_weight[bin_labels == t] = weight[t]
    return WeightedRandomSampler(samples_weight, len(samples_weight))

# Create sampler and loader
train_sampler = stratified_sampler(train_dataset[:]["arr_1"].view(-1))
train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=128)

# Initialize model
vocab_size = AminoAcidTokenizer("iupac1").vocab_size
model = LSTMModel(vocab_size, input_size=64, hidden_size=256, num_layers=3,
                  hidden_dropout=0.25)

# Init callbacks
stop_clbk = EarlyStopping(patience=3, verbose=1)
save_clbk = ModelCheckpoint("results/3gb1/lstm_fitness/", verbose=1,
                            save_weights_only=True, prefix="design0")
# Cumbersome, but required to ensure weights get saved properly.
# How do we ensure that the model (and its updated weights) are being used
# everytime we are sampling the new batch?
save_clbk.set_model(model)

# Construct loss function and optimizer
# NOTE: We use MSE loss because although there may be outliers in terms of the
# variant fitness score (few "great" variants), we still care about them. That
# is to say, we want the model to "learn" the features that make those protein
# variants more fit (according to their y score) than other variants.
criterion = torch.nn.MSELoss(reduction="mean")
optimizer = AdamW(model.parameters(), lr=1e-3)

print(f"Train on {len(train_idx)}, validate on {len(val_idx)}...")
# PSEUDOCODE: Until the convergence criteria is not met, i.e. the acqusition
#             function didn't change (usually modelled by KL(p || q))
# PSEUDOCODE: Update model checkpoint prefix such that it is design{idx}
for epoch in range(15):
    # Training loop
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        # Move feats/labels to gpu device (if available)
        feats, labels = [arr.to(device) for arr in batch.values()]
        optimizer.zero_grad()                   # zero gradients
        train_y_pred = model(feats)             # forward pass through model
        loss = criterion(train_y_pred, labels)  # compute loss
        loss.backward()                         # compute gradients
        optimizer.step()                        # update params/weights

        # Compute squared error (SE) across whole batch
        batch_loss = loss.item() * len(labels)
        train_loss += batch_loss
        # Print loss (for every kth batch)
        if (batch_idx+1) % 5 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, (batch_idx+1) * len(labels), len(train_loader.dataset),
                100. * (batch_idx+1) / len(train_loader), loss.item()))
    # MSE over all training examples
    print("====> Epoch: {} Train (avg) loss: {:.4f}".format(
        epoch, train_loss / len(train_loader.dataset)))

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_batch in val_loader:
            # Move feats/labels to gpu device (if available)
            val_feats, val_labels = [arr.to(device) for arr in val_batch.values()]
            val_y_pred = model(val_feats)
            loss = criterion(val_y_pred, val_labels)
            val_loss += loss.item() * len(val_labels)
    val_loss /= len(val_loader.dataset)
    print(f"====> Epoch: {epoch} Test (avg) loss: {val_loss:.4f}")

    # Stop training (based off val loss) and save (top k) ckpts
    save_clbk.on_epoch_end(epoch, logs={"val_loss": val_loss})
    should_stop = stop_clbk.on_epoch_end(epoch, logs={"val_loss": val_loss})
    if should_stop:
        break
