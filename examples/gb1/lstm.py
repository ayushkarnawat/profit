"""Train GB1 LSTM oracle."""

import os
import time
import multiprocessing as mp
import pandas as pd

import torch
from torch import optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from profit.dataset.splitters import split_method_dict
from profit.models.pytorch.lstm import LSTMModel
from profit.utils.data_utils.tokenizers import AminoAcidTokenizer
from profit.utils.training_utils.pytorch.callbacks import EarlyStopping
from profit.utils.training_utils.pytorch.callbacks import ModelCheckpoint
from profit.utils.training_utils.pytorch import losses as L

from data import load_dataset


timestep = time.strftime("%Y-%b-%d-%H:%M:%S", time.gmtime())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
splits = ["train", "valid"]
# splits = ["train"] if args.train_size > 0 else []
# splits += ["valid"] if args.valid_size > 0 else []
# splits += ["test"] if args.test_size > 0 else []

# Preprocess + load the dataset
dataset = load_dataset("lstm", "primary", labels="Fitness", num_data=-1,
                       filetype="mdb", as_numpy=False, vocab="aa20")

# Stratify train/val/test sets s.t. the target labels are equally represented in
# each subset. Each subset will have the same ratio of low/mid/high variants in
# each batch as the full dataset. See: https://discuss.pytorch.org/t/29907/2
_dataset = dataset[:]["arr_0"]
_labels = dataset[:]["arr_1"].view(-1)
# Create subset indicies
subset_idx = split_method_dict["stratified"]().train_valid_test_split(
    dataset=_dataset, labels=_labels.tolist(), frac_train=0.9,
    frac_val=0.1, frac_test=0., return_idxs=True, n_bins=5)
stratified = {split: Subset(dataset, sorted(idx))
              for split, idx in zip(splits, subset_idx)}

# Compute sample weight (each sample should get its own weight)
def stratified_sampler(labels: torch.Tensor,
                       nbins: int = 10) -> WeightedRandomSampler:
    bin_labels = torch.tensor(pd.qcut(labels.tolist(), q=nbins,
                                      labels=False, duplicates="drop"))
    class_sample_count = torch.tensor(
        [(bin_labels == t).sum() for t in torch.unique(bin_labels, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.zeros_like(labels)
    for t in torch.unique(bin_labels):
        samples_weight[bin_labels == t] = weight[t]
    return WeightedRandomSampler(samples_weight, len(samples_weight))

# Create sampler (only needed for train)
sampler = stratified_sampler(stratified["train"][:]["arr_1"].view(-1))

# Init model
vocab_size = AminoAcidTokenizer("aa20").vocab_size
model = LSTMModel(vocab_size, input_size=64, hidden_size=128, num_layers=2,
                  num_outputs=2, hidden_dropout=0.25)

# Init callbacks
# NOTE: Must set model (within save_clbk) to ensure weights get saved
stop_clbk = EarlyStopping(patience=5, verbose=1)
save_clbk = ModelCheckpoint(os.path.join("bin/3gb1/lstm", timestep),
                            monitor="val_loss",
                            verbose=1,
                            save_weights_only=True)
save_clbk.set_model(model)

# Init callbacks
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

step = 0
epochs = 50
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
            # Move data (sequence encoded) to GPU (if available)
            data, target = [arr.to(device) for arr in batch.values()]
            batch_size = data.size(0)

            # Forward pass
            pred = model(data)
            # Loss calculation
            nll_loss = L.gaussian_nll_loss(pred, target, reduction="sum")
            summed_loss += nll_loss.item()
            loss = nll_loss / batch_size
            # Compute gradients and update params/weights
            if split == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1

            # Bookkeeping (batch)
            if it % 5 == 0 or it+1 == len(data_loader):
                print("{} Batch {:04d}/{:d} ({:.2f}%)\tLoss: {:.4f}".format(
                    split.upper(), it+1, len(data_loader),
                    100. * ((it+1)/len(data_loader)), loss.item()))

        # Bookkeeping (epoch)
        avg_loss = summed_loss / len(data_loader.dataset)
        print("{} Epoch {}/{}, Average NLL loss: {:.4f}".format(
            split.upper(), epoch, epochs, avg_loss))

        # Stop training (based off val loss) and save (top k) ckpts
        if split == "valid":
            save_clbk.on_epoch_end(epoch, logs={"val_loss": avg_loss})
            should_stop = stop_clbk.on_epoch_end(epoch, logs={"val_loss": avg_loss})
            if should_stop:
                break
    else:
        continue
    break
