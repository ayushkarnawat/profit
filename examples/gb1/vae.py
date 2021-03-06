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
[1] Kingma, Diederik P., and Max Welling. "Auto-encoding variational
    bayes." arXiv preprint arXiv:1312.6114 (2013).

[2] Brookes, David H., Hahnbeom Park, and Jennifer Listgarten.
    "Conditioning by adaptive sampling for robust design." arXiv
    preprint arXiv:1901.10060 (2019).

[3] Balancing VAE loss: https://stats.stackexchange.com/q/341954

[4] Bowman, Samuel R., et al. "Generating sentences from a continuous
    space." arXiv preprint arXiv:1511.06349 (2015).
"""

import os
import json
import time
import argparse

from collections import defaultdict
from multiprocessing import cpu_count

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

from profit.dataset.splitters import split_method_dict
from profit.models.torch.vae import SequenceVAE
from profit.utils.data_utils import VOCABS
from profit.utils.data_utils.tokenizers import AminoAcidTokenizer
from profit.utils.training_utils.torch import losses as L

from data import load_variants


def main(args):

    ts = time.strftime("%Y-%b-%d-%H:%M:%S", time.gmtime())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    splits = ["train"] if args.train_size > 0 else []
    splits += ["valid"] if args.valid_size > 0 else []
    splits += ["test"] if args.test_size > 0 else []

    # Preprocess + load the dataset
    dataset = load_variants("lstm", labels="Fitness", num_data=5000,
                            filetype="mdb", as_numpy=False, vocab=args.vocab)

    # Split train/val/test sets randomly
    _dataset = dataset[:]["arr_0"]
    _labels = dataset[:]["arr_1"].view(-1)
    # Create subset indicies
    subset_idx = split_method_dict["stratified"]().train_valid_test_split(
        dataset=_dataset, labels=_labels.tolist(), frac_train=args.train_size,
        frac_valid=args.valid_size, frac_test=args.test_size, return_idxs=True,
        n_bins=5)
    stratified = {split: Subset(dataset, sorted(idx))
                  for split, idx in zip(splits, subset_idx)}

    # Initialize model
    tokenizer = AminoAcidTokenizer(args.vocab)
    vocab_size = tokenizer.vocab_size
    seqlen = _dataset.size(1)
    model = SequenceVAE(seqlen, vocab_size, args.hidden_size,
                        args.latent_size).to(device)

    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Logging dir
    if args.tensorboard_logging:
        # NOTE: On MacOS, len(exp_name) < 255 to ensure os.makedirs() makes the
        # dir. This is due to the hard limit set by the filesystem. Therefore,
        # we remove non-essential vars from the name.
        remove_vars = ["train_size", "valid_size", "test_size", "print_every",
                       "tensorboard_logging", "logdir", "save_model_path",
                       "dumpdir"]
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
                shuffle=split == "train",
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
                nll_loss, kl_loss, kl_weight = L.elbo_loss(pred, data, mu, \
                    logvar, args.anneal_function, step, args.k, args.x0)
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
                    # Apply softmax to convert logits -> probs (across each
                    # amino acid) to reconstruct seqs. NOTE: If we have certain
                    # tokens in our vocab that we do not want to include in the
                    # reconstruction (i.e. <pad>/<unk>), we have to remove their
                    # softmax prediction. Otherwise, the model might generate
                    # sequences with some of those vocab included.
                    recon_seqs = torch.argmax(F.softmax(pred, dim=-1), dim=-1)

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
                if not os.path.exists(os.path.join(args.dumpdir, ts)):
                    os.makedirs(os.path.join(args.dumpdir, ts))
                with open(os.path.join(args.dumpdir, ts, f"valid_E{epoch:04d}.json"),
                          "w") as dump_file:
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
    parser.add_argument("--train_size", type=float, default=1.0, nargs='?', const=1)
    parser.add_argument("--valid_size", type=float, default=0., nargs='?', const=1)
    parser.add_argument("--test_size", type=float, default=0., nargs='?', const=1)

    parser.add_argument("-ep", "--epochs", type=int, default=50)
    parser.add_argument("-bs", "--batch_size", type=int, default=32)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)

    # NOTE: Instead of defining the embedding_size (usually same as num_vocab
    # in the dictionary), we instead ask what (pre-defined) vocabulary to use.
    # parser.add_argument("-eb", "--embedding_size", type=int, default=300)
    parser.add_argument("-vb", "--vocab", type=str, default="aa20")
    parser.add_argument("-hs", "--hidden_size", type=int, default=50)
    parser.add_argument("-ls", "--latent_size", type=int, default=20)

    parser.add_argument("-af", "--anneal_function", type=str, default="logistic")
    parser.add_argument("-k", "--k", type=float, default=0.0025)
    parser.add_argument("-x0", "--x0", type=int, default=2500)

    parser.add_argument("-v", "--print_every", type=int, default=5)
    parser.add_argument("-tb", "--tensorboard_logging", action="store_true")
    parser.add_argument("-log", "--logdir", type=str, default="logs/3gb1/vae")
    parser.add_argument("-bin", "--save_model_path", type=str, default="bin/3gb1/vae")
    parser.add_argument("-dump", "--dumpdir", type=str, default="dumps/3gb1/vae")

    args = parser.parse_args()
    args.vocab = args.vocab.lower()
    if args.anneal_function == "None":
        args.anneal_function = None
    else:
        args.anneal_function = args.anneal_function.lower()

    assert args.vocab in VOCABS
    assert args.anneal_function in ["logistic", "linear", None]

    main(args)
