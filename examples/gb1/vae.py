r"""Train GB1 variational autoencoder (VAE) model.

The model encodes information about the input `x` into a latent variable
`z` (through `mu` and `logvar`) so that the decoder network can generate
the reconstructed output `x'` (such that `x \approx x'`) [1].

The latent variable `z` allows us to generate samples from a continuous
space `z \in \mathbb{R}^N` onto the data `x \in \mathbb{R}^L`, where `N`
is the size of the latent vector and `L` is the length of the sequence.
Our hope is that this generative model, `p(x|z)`, can be trained to
approximate the unknown underlying data distribution `p_d(x)` well using
a small set of realizations `X` drawn from that distribution.

For the purpose of this task, we denote by `\theta^{(0)}` the parameters
of the generative model after it has been fit to the data `X`, which
yields the prior density `p(x|\theta^{(0)})`. This prior can be used to
sample new points to explore [2].

Training:
---------
During training, when the loss is being optimized, the KL term decreases
quickly, which prevents the reconstruction loss from decreasing [3].
This forces the latent vector `z \sim q(z|x)`, which means the model
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

import os
import json
import math
import time
import argparse

from collections import defaultdict
from multiprocessing import cpu_count

import pandas as pd
from tensorboardX import SummaryWriter

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from profit.dataset.splitters import split_method_dict
from profit.models.pytorch.vae import SequenceVAE
from profit.utils.data_utils.tokenizers import AminoAcidTokenizer
from profit.utils.training_utils.pytorch.optimizers import AdamW

from data import load_dataset


def main(args):

    ts = time.strftime("%Y-%b-%d-%H:%M:%S", time.gmtime())
    splits = ["train", "valid"] + (["test"] if args.test else [])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    # Preprocess + load the dataset
    dataset = load_dataset("lstm", "primary", labels="Fitness", num_data=-1,
                           filetype="mdb", as_numpy=False)

    # Stratify train/val/test sets such that the target labels are equally
    # represented in each subset. Each subset will have the same ratio of
    # low/mid/high variants in each batch as the full dataset.
    # See: https://discuss.pytorch.org/t/29907/2
    _dataset = dataset[:]["arr_0"]
    _labels = dataset[:]["arr_1"].view(-1)
    # Create subset indicies
    subset_idx = split_method_dict["stratified"]().train_valid_test_split(
        _dataset, labels=_labels.tolist(), frac_train=0.8, frac_val=0.2,
        frac_test=0., return_idxs=True, n_bins=10)
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

    # Create sampler
    sampler = stratified_sampler(stratified["train"][:]["arr_1"].view(-1))

    # Initialize model
    tokenizer = AminoAcidTokenizer(args.vocab)
    vocab_size = tokenizer.vocab_size
    model = SequenceVAE(vocab_size, args.hidden_size_1, args.hidden_size_2,
                        args.latent_size).to(device)

    # Anneal KL-divergence term, see: https://arxiv.org/abs/1511.06349
    def kl_anneal_function(anneal_function, step, k=0.0025, x0=2500):
        if anneal_function == "logistic":
            return float(1/(1+math.exp(-k*(step-x0))))
        elif anneal_function == "linear":
            return min(1, step/x0)

    # Construct loss function
    def loss_fn(pred, target, mu, logvar, anneal_function, step, k, x0):
        """Compute variance of evidence lower bound (ELBO) loss.

        NOTE: The pred values should be logits (raw values). That is, no
        softmax/log softmax should be applied to the outputs. This is
        because F.cross_entropy() applies a F.log_softmax() internally
        before computing the negative log likelihood using F.nll_loss().
        """
        # Reconstruction loss
        # pred=(N,s,b), target=(N,s), where N=batch_size, s=seqlen, b=vocab_size
        pred = pred.permute(0, 2, 1) # Must be (N,b,s) for F.cross_entropy
        nll_loss = F.cross_entropy(pred, target, reduction="sum")

        # KL Divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_weight = kl_anneal_function(anneal_function, step, k, x0)

        return nll_loss, kl_loss, kl_weight

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Logging dir
    if args.tensorboard_logging:
        # NOTE: On MacOS, len(exp_name) < 255 to ensure os.makedirs() makes the
        # dir. This is due to the hard limit set by the filesystem. Therefore,
        # we remove non-essential vars from the name.
        remove_vars = ["logdir", "print_every", "save_model_path",
                       "tensorboard_logging", "test"]
        params = "__".join([f"{key}={value}" for key, value in vars(args).items()
                            if key not in remove_vars])
        exp_name = params + f"__time={ts}"
        writer = SummaryWriter(os.path.join(args.logdir, exp_name))
        writer.add_text("model", str(model))
        writer.add_text("args", str(args))
        writer.add_text("ts", ts)

    save_model_path = os.path.join(args.save_model_path, ts)
    os.makedirs(save_model_path)

    step = 0
    for epoch in range(args.epochs):
        for split in splits:
            tracker = defaultdict(tensor)
            data_loader = DataLoader(
                dataset=stratified[split],
                batch_size=args.batch_size,
                sampler=sampler if split == "train" else None,
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )

            # Enable/disable dropout
            model.train() if split == "train" else model.eval()

            for it, batch in enumerate(data_loader):
                # Move data (sequence encoded) to GPU (if available)
                data = batch["arr_0"].long().to(device)
                # One-hot encode (see: https://discuss.pytorch.org/t/507/34)
                batch_size, seqlen = data.size()
                onehot = torch.zeros(batch_size, seqlen, vocab_size)
                onehot.scatter_(2, torch.unsqueeze(data, 2), 1)

                # Forward pass
                pred, mu, logvar, z = model(onehot)
                # Loss calculation
                nll_loss, kl_loss, kl_weight = loss_fn(pred, data, mu, logvar, \
                    args.anneal_function, step, args.k, args.x0)
                loss = (nll_loss + kl_weight * kl_loss) / batch_size
                # Compute gradients and update params/weights
                if split == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1

                # Bookkeeping (batch)
                tracker["ELBO"] = torch.cat((tracker["ELBO"], tensor([loss.data])))
                if args.tensorboard_logging:
                    writer.add_scalar(f"{split.upper()}/ELBO", loss.data,
                                      epoch * len(data_loader) + it)
                    writer.add_scalar(f"{split.upper()}/NLL Loss",
                                      nll_loss.data / batch_size,
                                      epoch * len(data_loader) + it)
                    writer.add_scalar(f"{split.upper()}/KL Loss",
                                      kl_loss.data / batch_size,
                                      epoch * len(data_loader) + it)
                    writer.add_scalar(f"{split.upper()}/KL Weight",
                                      kl_weight, epoch*len(data_loader) + it)

                if it % args.print_every == 0 or it+1 == len(data_loader):
                    print("{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tNLL-Loss: "
                          "{:.4f}\tKL-Loss: {:.4f}\tKL-Weight: {:.4f}".format(
                              split.upper(), epoch, it * len(data),
                              len(data_loader.dataset), 100. * it / len(data_loader),
                              loss.item(), nll_loss.item() / batch_size,
                              kl_loss.item() / batch_size, kl_weight))

                if split == "valid":
                    # Apply softmax to convert logits -> prob to reconstruct seqs
                    recon_seqs = torch.argmax(F.softmax(pred), dim=-1)

                    if "recon_seqs" not in tracker:
                        tracker["recon_seqs"] = list()
                    if "target_seqs" not in tracker:
                        tracker["target_seqs"] = list()
                    for rseq, tseq in zip(recon_seqs.numpy(), data.numpy()):
                        tracker["recon_seqs"] += ["".join(tokenizer.decode(rseq))]
                        tracker["target_seqs"] += ["".join(tokenizer.decode(tseq))]
                    tracker["z"] = torch.cat((tracker["z"], z.data), dim=0)

            # Bookkeeping (epoch)
            if args.tensorboard_logging:
                writer.add_scalar(f"{split.upper()}-Epoch/ELBO",
                                  torch.mean(tracker["ELBO"]), epoch)
            print("{} Epoch {}/{}, Average ELBO loss: {:.4f}".format(
                split.upper(), epoch, args.epochs, torch.mean(tracker["ELBO"])))

            # Save dump of all valid seqs and the encoded latent space
            if split == "valid":
                dump = {
                    "recon_seqs": tracker["recon_seqs"],
                    "target_seqs": tracker["target_seqs"],
                    "z": tracker["z"].tolist()
                }
                if not os.path.exists(os.path.join("dumps", ts)):
                    os.makedirs(os.path.join("dumps", ts))
                with open(os.path.join("dumps", ts, f"valid_E{epoch:04d}.json"), "w") as dump_file:
                    json.dump(dump, dump_file)

            # Save checkpoint
            if split == "train":
                checkpoint_path = os.path.join(save_model_path, f"E{epoch:04d}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Model saved at {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("--data_dir", type=str, default="data")
    # parser.add_argument("--max_sequence_length", type=int, default=60)
    parser.add_argument("--test", action="store_true")

    parser.add_argument("-ep", "--epochs", type=int, default=50)
    parser.add_argument("-bs", "--batch_size", type=int, default=16)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)

    # Instead of defining the embedding_size (usually same as num_vocab in your
    # dictionary), we instead ask what (pre-defined) vocabulary to use.
    # parser.add_argument("-eb", "--embedding_size", type=int, default=300)
    parser.add_argument("-vb", "--vocab", type=str, default="iupac1")
    parser.add_argument("-hs1", "--hidden_size_1", type=int, default=64)
    parser.add_argument("-hs2", "--hidden_size_2", type=int, default=32)
    parser.add_argument("-ls", "--latent_size", type=int, default=10)

    parser.add_argument("-af", "--anneal_function", type=str, default="logistic")
    parser.add_argument("-k", "--k", type=float, default=0.0025)
    parser.add_argument("-x0", "--x0", type=int, default=2500)

    parser.add_argument("-v", "--print_every", type=int, default=5)
    parser.add_argument("-tb", "--tensorboard_logging", action="store_true")
    parser.add_argument("-log", "--logdir", type=str, default="logs")
    parser.add_argument("-bin", "--save_model_path", type=str, default="bin")

    args = parser.parse_args()
    args.anneal_function = args.anneal_function.lower()

    assert args.vocab in ["iupac1", "iupac3"]
    assert args.anneal_function in ["logistic", "linear"]

    main(args)
