"""Train GB1 variational autoencoder (VAE) model.

The model encodes information about the input `x` into a latent variable
`z` (through `mu` and `logvar`) so that the decoder network can generate
the reconstructed output `x'` (such that `x \\approx x'`) [1].

The latent variable `z` allows us to generate samples from a continuous
space `z \\in \\mathbb{R}^N` onto the data `x \\in \\mathbb{R}^L`,
where `N` is the size of the latent vector and `L` is the length of the
sequence. Our hope is that this generative model, `p(x|z)`, can be
trained to approximate the unknown underlying data distribution `p_d(x)`
well using a small set of realizations `X` drawn from that distribution.

For the purpose of this task, we denote by `\\theta^{(0)}` the parameters
of the generative model after it has been fit to the data `X`, which
yields the prior density `p(x|\\theta^{(0)})`. This prior can be used to
sample new points to explore [2].

Training:
---------
During training, when the loss is being optimized, the KL term decreases
quickly, which prevents the reconstruction loss from decreasing [3].
This forces the latent vector `z \\sim q(z|x)`, which means the model
cannot differentiate between samples drawn from the normal gaussian
`\mathcal{N}(0,I)` and the actual data. Essentially, this means that
the model did not learn the "core" features of the underlying data.

To fix this problem, we can anneal the KL-divergence term such that the
network goes from a vanilla autoencoder to a variational one. At the
start of training, the weight is set to 0, so that the model learns to
encode as much info into `z` as it can. Then, as training progresses,
we gradually increase this weight, forcing the model to smooth out its
encodings and pack them into the prior. We increase this weight until
it reaches 1, at which point the weighted cost function is equivalent
to the true variational lower bound [4].

References:
-----------
-[1] VAE paper: https://arxiv.org/abs/1312.6114
-[2] CbAS paper: https://arxiv.org/abs/1901.10060
-[3] Balancing VAE loss: https://stats.stackexchange.com/q/341954
-[4] Sentence-VAE paper: https://arxiv.org/abs/1511.06349
"""

import numpy as np
import pandas as pd

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from profit.dataset.splitters import split_method_dict
from profit.models.pytorch.vae import SequenceVAE
from profit.utils.data_utils.tokenizers import AminoAcidTokenizer
from profit.utils.training_utils.pytorch.optimizers import AdamW

from data import load_dataset


# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocess + load the dataset
dataset = load_dataset("lstm", "primary", labels="Fitness", num_data=-1,
                       filetype="mdb", as_numpy=False)

# Stratify the dataset into train/val sets
# NOTE: We use a stratified sampler to split the target labels equally into each
# subset. That is, both the train and validation datasets will have the same
# ratio of low/mid/high fitness variants as the full dataset in each batch.
# See: https://discuss.pytorch.org/t/29907/2
_dataset = dataset[:]["arr_0"]
_labels = dataset[:]["arr_1"].view(-1)
# Create subset indicies
train_idx, val_idx = split_method_dict["stratified"]().train_valid_split(
    _dataset, labels=_labels.tolist(), frac_train=0.8, frac_val=0.2,
    return_idxs=True, n_bins=10)
train_dataset = Subset(dataset, sorted(train_idx))
val_dataset = Subset(dataset, sorted(val_idx))

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
train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=128)


# Initialize model
vocab_size = AminoAcidTokenizer('iupac1').vocab_size
model = SequenceVAE(vocab_size, h_dim1=64, h_dim2=32, latent_size=10).to(device)

# Anneal KL-divergence term, see: https://arxiv.org/abs/1511.06349
def kl_anneal_function(anneal_function, step, k=0.0025, x0=2500):
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)

# Construct loss function
def loss_fn(logp, target, mu, logvar, anneal_function, step, k, x0):
    # Reconstruction loss
    # logp=(N,s,b), target=(N,s), where N=batch_size, s=seqlen, b=vocab_size
    logp = logp.permute(0, 2, 1) # Must be (N,b,s) for F.nll_loss
    NLL_loss = F.nll_loss(logp, target, reduction="sum")

    # KL Divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KL_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KL_weight = kl_anneal_function(anneal_function, step, k, x0)

    return NLL_loss, KL_loss, KL_weight

optimizer = AdamW(model.parameters(), lr=1e-3)

print(f"Train on {len(train_idx)}, validate on {len(val_idx)}...")
step = 0
for epoch in range(1, 201):
    # Training loop
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        # Move data (sequence encoded) to gpu device (if available)
        data = batch["arr_0"].long().to(device)
        # One-hot encode (see: https://discuss.pytorch.org/t/507/34)
        batch_size, seqlen = data.size()
        onehot = torch.zeros(batch_size, seqlen, vocab_size)
        onehot.scatter_(2, torch.unsqueeze(data, 2), 1)

        optimizer.zero_grad()
        logp, mu, logvar = model(onehot)
        NLL_loss, KL_loss, KL_weight = loss_fn(logp, data, mu, logvar, \
            anneal_function="logistic", step=step, k=0.0025, x0=2500)
        train_loss += (NLL_loss + KL_weight * KL_loss).item()
        loss = (NLL_loss + KL_weight * KL_loss) / batch_size
        loss.backward()
        optimizer.step()
        step += 1

        # Print loss (every kth batch)
        if batch_idx % 2 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tNLL-Loss: "
                  "{:.6f}\tKL-Loss: {:.6f}\tKL-Weight: {:.3f}".format(
                      epoch, batch_idx * len(data), len(train_loader.dataset),
                      100. * batch_idx / len(train_loader), loss.item(),
                      NLL_loss.item() / batch_size, KL_loss.item() / batch_size,
                      KL_weight))
    # Average loss over all training examples
    print(train_loss)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for _, batch in enumerate(val_loader):
            # One-hot encode (see: https://discuss.pytorch.org/t/507/34)
            data = batch["arr_0"].long().to(device)
            batch_size, seqlen = data.size()
            onehot = torch.zeros(batch_size, seqlen, vocab_size)
            onehot.scatter_(2, torch.unsqueeze(data, 2), 1)

            logp, mu, logvar = model(onehot)
            NLL_loss, KL_loss, KL_weight = loss_fn(logp, data, mu, logvar, \
                anneal_function="logistic", step=step, k=0.0025, x0=2500)
            val_loss += (NLL_loss + KL_weight * KL_loss).item()
    val_loss /= len(val_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(val_loss))
