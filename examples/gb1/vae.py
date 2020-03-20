import pandas as pd

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from profit.dataset.splitters import split_method_dict
from profit.models.pytorch.vae import SequenceVAE
from profit.utils.training_utils.pytorch.optimizers import AdamW
from profit.utils.training_utils.pytorch.callbacks import EarlyStopping
from profit.utils.training_utils.pytorch.callbacks import ModelCheckpoint

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
train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=128)


# Initialize model
seqlen = _dataset.size(1)
model = SequenceVAE(seqlen, h_dim1=256, h_dim2=128, latent_size=2).to(device)


# # Init callbacks
# stop_clbk = EarlyStopping(patience=3, verbose=1)
# save_clbk = ModelCheckpoint("results/3gb1/vae_fitness/", verbose=1,
#                             save_weights_only=True, prefix="design0")
# # Cumbersome, but required to ensure weights get saved properly.
# # How do we ensure that the model (and its updated weights) are being used
# # everytime we are sampling the new batch?
# save_clbk.set_model(model)


# Construct loss function and optimizer
def criterion(recon_x, x, mu, logvar):
    """Reconstruction + KL divergence losses summed over all elements.
    See: https://github.com/Lasagne/Recipes/issues/54 for potential fix.
    The MSE Loss is exploding...maybe because the input data is not normalized? Not sure
    """
    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    print(MSE)
    # see Appendix B from VAE paper: https://arxiv.org/abs/1312.6114
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

optimizer = AdamW(model.parameters(), lr=1e-3)


def train(epoch):
    """Training loop."""
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        # Move data (sequence encoded) to gpu device (if available)
        data = batch["arr_0"].to(device)
        optimizer.zero_grad()                           # zero gradients
        recon_batch, mu, logvar = model(data)           # forward pass through model
        loss = criterion(recon_batch, data, mu, logvar) # compute loss
        loss.backward()                                 # compute gradients
        optimizer.step()                                # update params/weights

        # Compute reconstruction + kl loss
        train_loss += loss.item()
        # Print loss (for every kth batch)
        if batch_idx % 2 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    # Average loss over all training examples
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test():
    """Test loop."""
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            data = batch["arr_0"].to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += criterion(recon_batch, data, mu, logvar).item()
            # # Visually compare the first n=8 samples at each epoch to show how
            # # well the model has been learning the latent representation `z`
            # if i == 0:
            #     n = min(data.size(0), 8)
            #     idxs = torch.where(recon_batch[:n] != data[:n])
            #     print(idxs)
            #     comparison = torch.cat([data[:n],
            #                            recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            #     save_image(comparison.cpu(),
            #              'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(val_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


print(f"Train on {len(train_idx)}, validate on {len(val_idx)}...")
for epoch in range(1, 51):
    train(epoch)
    test()
 