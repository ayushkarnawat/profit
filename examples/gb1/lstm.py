"""Train GB1 LSTM oracle."""

import math
import multiprocessing as mp

import pandas as pd

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from profit.dataset.splitters import split_method_dict
from profit.models.pytorch.lstm import LSTMModel
from profit.utils.data_utils.tokenizers import AminoAcidTokenizer
from profit.utils.training_utils.pytorch.callbacks import EarlyStopping
from profit.utils.training_utils.pytorch.callbacks import ModelCheckpoint
from profit.utils.training_utils.pytorch.optimizers import AdamW

from data import load_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
splits = ["train", "valid"]
# splits = ["train"] if args.train_size > 0 else []
# splits += ["valid"] if args.valid_size > 0 else []
# splits += ["test"] if args.test_size > 0 else []

# Preprocess + load the dataset
dataset = load_dataset("lstm", "primary", labels="Fitness", num_data=-1,
                       filetype="mdb", as_numpy=False, vocab="aa20")

# Stratify train/val/test sets s.t. the target labels are equally represented
# in each subset. Each subset will have the same ratio of low/mid/high variants
# in each batch as the full dataset. See: https://discuss.pytorch.org/t/29907/2
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
model = LSTMModel(vocab_size, input_size=64, hidden_size=256, num_layers=3,
                  num_outputs=2, hidden_dropout=0.25)

# Init callbacks
# NOTE: Must set model (within save_clbk) to ensure weights get saved
stop_clbk = EarlyStopping(patience=5, verbose=1)
save_clbk = ModelCheckpoint("bin/3gb1/lstm/", monitor="val_loss", verbose=1,
                            save_weights_only=True, prefix="")
save_clbk.set_model(model)

# Construct loss function
def loss_fn(pred, target, reduction="mean"):
    """Compute NLL loss of a `N(\\mu,\\sigma^2)` gaussian distribution.

    Usually, when computing the loss between two continous values (for
    linear regression tasks), we use the mean square error (MSE) loss
    because, although there may be outliers in terms of the fitness
    score (aka very few "great" variants), we want to give more weight
    to the larger differences. That is to say, we want the model to
    "learn" the features that make those protein variants more "fit"
    (based off their fitness score) than other variants.

    However, we want to find the ideal paramater `\\theta` that minimizes
    `y` from the likelihood function `p(y|x, \\theta)`. It can be easily
    shown that minimizing the NLL of our data with respect to `\\theta`
    is equivalent to minimizing the MSE between the observed `y` and our
    prediction. That is, the `\\argmin(NLL)` = `\\argmin(MLE)`.

    See: http://willwolf.io/2017/05/18/minimizing_the_negative_log_likelihood_in_english/

    Params:
    -------
    pred: torch.Tensor, size=(N,2)
        Prediction of the mean and variance of the response variable.

    target: torch.Tensor, size=(N)
        Ground truth (mean) value.

    reduction: str, default="mean"
        Specifies the reduction to apply to the output. If "mean", the
        sum of the output will be divided by the number of elements in
        the output. If "sum", the output will be summed.
    """
    if pred.size(0) != target.size(0):
        raise ValueError(f"Sizes do not match ({pred.size(0)} != {target.size(0)}).")
    N = pred.size(0)
    mu = pred[:, 0]
    # We use softplus and add 1e-6 to avoid divide by 0 error
    var = F.softplus(pred[:, 1]) + 1e-6
    logvar = torch.log(var)
    target = target.squeeze(1)
    if reduction == "mean":
        return 0.5 * torch.log(tensor([math.tau])) + 0.5 * torch.mean(logvar) \
            + 0.5 * torch.mean(torch.square(target - mu) / var)
    return 0.5 * N * torch.log(tensor([math.tau])) + 0.5 * torch.sum(logvar) \
        + torch.sum(torch.square(target - mu) / (2 * var))

optimizer = AdamW(model.parameters(), lr=1e-3)

step = 0
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
            # Move data (sequence encoded) to GPU (if available)
            data, target = [arr.to(device) for arr in batch.values()]
            batch_size = data.size(0)

            # Forward pass
            pred = model(data)
            # Loss calculation
            nll_loss = loss_fn(pred, target, reduction="sum")
            summed_loss += nll_loss.item()
            loss = nll_loss / batch_size
            # Compute gradients and update params/weights
            if split == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1

            # Bookkeeping (batch)
            if it % 2 == 0 or it+1 == len(data_loader):
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
