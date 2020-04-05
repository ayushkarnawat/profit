"""Train 3D-Embedded Graph Convolution Network (EGCN) oracle.

NOTE: This model is not recommened to be run on CPU as the model
performs convolutional kernels computations over a large amount of 3D
graphical data. Thus, it requires a lot of compute.
"""

import multiprocessing as mp
import pandas as pd

import torch
from torch import optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from profit.dataset.splitters import split_method_dict
from profit.models.pytorch.egcn import EmbeddedGCN
from profit.utils.training_utils.pytorch.callbacks import EarlyStopping
from profit.utils.training_utils.pytorch.callbacks import ModelCheckpoint
from profit.utils.training_utils.pytorch import losses as L

from data import load_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits = ["train", "valid"]

# Preprocess + load the dataset
dataset = load_dataset("egcn", "tertiary", labels="Fitness", num_data=5,
                       filetype="mdb", as_numpy=False)

# Stratify train/val/test sets such that the target labels are equally
# represented in each subset. Each subset will have the same ratio of
# low/mid/high variants in each batch as the full dataset.
# See: https://discuss.pytorch.org/t/29907/2
_dataset = torch.Tensor(len(dataset), 1)
_labels = dataset[:]["arr_3"].view(-1).tolist()
# Create subset indicies
subset_idx = split_method_dict["stratified"]().train_valid_test_split(
    _dataset, _labels, frac_train=0.8, frac_val=0.2, frac_test=0.,
    return_idxs=True, n_bins=5)
stratified = {split: Subset(dataset, sorted(idx))
              for split, idx in zip(splits, subset_idx)}

# Compute sample weight (each sample should get its own weight)
def stratified_sampler(labels: torch.Tensor,
                       nbins: int = 10) -> WeightedRandomSampler:
    if torch.unique(labels).size(0) > 1:
        bin_labels = torch.tensor(pd.qcut(labels.tolist(), q=nbins,
                                          labels=False, duplicates="drop"))
        class_sample_count = torch.tensor(
            [(bin_labels == t).sum() for t in torch.unique(bin_labels, sorted=True)])
        weight = 1. / class_sample_count.float()
        samples_weight = torch.zeros_like(labels)
        for t in torch.unique(bin_labels):
            samples_weight[bin_labels == t] = weight[t]
    else:
        samples_weight = torch.rand(1).repeat(len(labels))
    return WeightedRandomSampler(samples_weight, len(samples_weight))

# Create sampler (only needed for train)
sampler = stratified_sampler(stratified["train"][:]["arr_3"].view(-1))

# Initialize model
num_atoms, num_feats = dataset[0]["arr_0"].shape
model = EmbeddedGCN(num_atoms, num_feats, num_outputs=2, num_layers=1,
                    units_conv=32, units_dense=64)

# Initialize callbacks
# NOTE: Must set model (within save_clbk) to ensure weights get saved
stop_clbk = EarlyStopping(patience=5, verbose=1)
save_clbk = ModelCheckpoint("bin/3gb1/egcn/", monitor="val_loss", verbose=1,
                            save_weights_only=True, prefix="")
save_clbk.set_model(model)

# Initialize optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

epochs = 15
for epoch in range(1, epochs+1):
    for split in splits:
        summed_loss = 0
        data_loader = DataLoader(
            dataset=stratified[split],
            batch_size=32,
            sampler=sampler if split == "train" else None,
            num_workers=mp.cpu_count(),
            pin_memory=torch.cuda.is_available()
        )

        # Enable/disable dropout
        model.train() if split == "train" else model.eval()

        for it, batch in enumerate(data_loader):
            # Move data to GPU (if available)
            atoms, adjms, dists, target = [arr.to(device) for arr in batch.values()]
            batch_size = atoms.size(0)

            # Forward pass
            pred = model([atoms, adjms, dists])
            # Loss calculation
            nll_loss = L.gaussian_nll_loss(pred, target, reduction="sum")
            summed_loss += nll_loss.item()
            loss = nll_loss / batch_size
            # Compute gradients and update params/weights
            if split == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Bookkeeping (batch)
            if it % 2 == 0 or it+1 == len(data_loader):
                print("{} Batch {:04d}/{:d} ({:.2f}%)\tLoss: {:.4f}".format(
                    split.upper(), it+1, len(data_loader),
                    100. * ((it+1)/len(data_loader)), loss.item()))

        # Bookkeeping (epoch)
        avg_loss = summed_loss / len(data_loader.dataset)
        print("{} Epoch {}/{}, Average NLL (MSE) loss: {:.4f}".format(
            split.upper(), epoch, epochs, avg_loss))

        # Stop training (based off val loss) and save (top k) ckpts
        if split == "valid":
            logs = {"val_loss": avg_loss}
            save_clbk.on_epoch_end(epoch, logs=logs)
            should_stop = stop_clbk.on_epoch_end(epoch, logs=logs)
            if should_stop:
                break
    else:
        continue
    break
