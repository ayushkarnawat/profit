"""Train (basic) densely-connected oracle."""

import os
import time
import multiprocessing as mp

import pandas as pd

import torch
from torch import optim
from torch.utils.data import DataLoader, Subset, TensorDataset, WeightedRandomSampler

from profit.dataset.splitters import split_method_dict
from profit.models.torch import SequenceOracle
from profit.utils.data_utils.tokenizers import AminoAcidTokenizer
from profit.utils.training_utils.torch import losses as L
from profit.utils.training_utils.torch.callbacks import ModelCheckpoint
from profit.utils.training_utils.torch.callbacks import EarlyStopping

from examples.gb1.data import load_dataset


timestep = time.strftime("%Y-%b-%d-%H:%M:%S", time.gmtime())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
splits = ["train", "valid"]

# Preprocess + load the dataset
dataset = load_dataset("lstm", "primary", labels="Fitness", num_data=-1,
                       filetype="mdb", as_numpy=False, vocab="aa20")
# Stratify train/val/test sets s.t. the target labels are equally represented in
# each subset. Each subset will have the same ratio of low/mid/high variants in
# each batch as the full dataset. See: https://discuss.pytorch.org/t/29907/2
_dataset = dataset[:]["arr_0"]
_labels = dataset[:]["arr_1"].view(-1)
# # Remove samples below a certain threshold
# high_idx = torch.where(_labels > _labels.mean())
# dataset = Subset(dataset, sorted(high_idx))
# _dataset = _dataset[high_idx]
# _labels = _labels[high_idx]

# Compute sample weights (each sample should get its own weight)
def sampler(labels: torch.Tensor,
            nbins: int = 10,
            stratify: bool = False) -> WeightedRandomSampler:
    discretize = pd.qcut if stratify else pd.cut
    bin_labels = torch.LongTensor(discretize(labels.tolist(), nbins,
                                             labels=False, duplicates="drop"))
    class_sample_count = torch.LongTensor(
        [(bin_labels == t).sum() for t in torch.arange(nbins)])
    weight = 1. / class_sample_count.float()
    sample_weights = torch.zeros_like(labels)
    for t in torch.unique(bin_labels):
        sample_weights[bin_labels == t] = weight[t]
    return WeightedRandomSampler(sample_weights, len(sample_weights))

# Compute sample weights and add to original dataset
weights = sampler(_labels, nbins=10, stratify=False).weights.type(torch.float)
dataset = TensorDataset(*dataset[:].values(), weights)

# Create subset indicies
subset_idx = split_method_dict["stratified"]().train_valid_test_split(
    dataset=_dataset, labels=_labels.tolist(), frac_train=0.9,
    frac_valid=0.1, frac_test=0.0, return_idxs=True, n_bins=10)
stratified = {split: Subset(dataset, sorted(idx))
              for split, idx in zip(splits, subset_idx)}

# Create stratified sampler (only needed for training)
train_sampler = sampler(stratified["train"][:][1].view(-1), stratify=True)

# Initialize model
tokenizer = AminoAcidTokenizer("aa20")
vocab_size = tokenizer.vocab_size
seqlen = stratified["train"][0][0].size(0)
model = SequenceOracle(seqlen, vocab_size, hidden_size=50, out_size=2)

# Initialize callbacks
# NOTE: Must set model (within save_clbk) to ensure weights get saved
stop_clbk = EarlyStopping(patience=5, verbose=1)
save_clbk = ModelCheckpoint(os.path.join("bin/3gb1/oracle", timestep),
                            monitor="val_loss",
                            verbose=1,
                            save_weights_only=True)
save_clbk.set_model(model)

# Initialize callbacks
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

epochs = 50
for epoch in range(1, epochs+1):
    for split in splits:
        summed_loss = 0
        data_loader = DataLoader(
            dataset=stratified[split],
            batch_size=32,
            sampler=train_sampler if split == "train" else None,
            num_workers=mp.cpu_count(),
            pin_memory=torch.cuda.is_available()
        )

        # Enable/disable dropout
        model.train() if split == "train" else model.eval()

        for it, batch in enumerate(data_loader):
            data = batch[0].long().to(device)
            target = batch[1].to(device)
            sample_weight = batch[2].to(device)
            # One-hot encode (see: https://discuss.pytorch.org/t/507/34)
            batch_size, seqlen = data.size()
            onehot = torch.zeros(batch_size, seqlen, vocab_size)
            onehot.scatter_(2, torch.unsqueeze(data, 2), 1)

            # Forward pass
            pred = model(onehot)
            # Loss calculation
            nll_loss = L.gaussian_nll_loss(pred, target, reduction="none")
            # Reweight nll_loss w/ sample weights
            nll_loss = (nll_loss * sample_weight).sum()
            summed_loss += nll_loss.item()
            loss = nll_loss / batch_size
            # Compute gradients and update params/weights
            if split == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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
