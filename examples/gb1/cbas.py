"""Weighted Optimization (CbAS).

This procedure aims to optimize the VAE is such a fashion that it
generates plausible sequences (similar to the ones in the search space),
and uses them to predict on the oracles.

Steps:
------
NOTE: For the VAE model, we use 5000 randomly sampled variants, which
are useful to generate new highly "plausible" variants, for each of the
next timesteps. We found that by using only the provided variants, the
generative model did not learn a good latent representation of the
sequences.
1. Train + save VAE (on whole dataset)
2. Train + save oracle(s) w/ early stopping (thru validation data)
3. Train + save SequenceGP (we denote this as GT)..not ideal but works
   as a proxy for actual predictions. This was tested on the validation
   set.
4. For t iterations
  - Gather samples from latent
      - Gather all possible samples
      - Remove samples that are not variants from the training set?
        this would ensure we only select seqs from the 160K samples)
  - Evaluate preds on oracles (avg them using balaji scheme if using
    multiple ones) and gather GT predictions
  - Compute weighting scheme
  - Track progress, in particular which is the best prediction. Also
    save, per iteration, the space (z) the values occupy and their
    prediction as a color value. Alternatively, one can make the x axis
    and y exis as the 4 mutated sequences, with the output of the
    optimization scheme returning the fitness value.
  - Retrain VAE so that it can generate new (and hopefully better)
    samples in the search space that are "worth exploring".

TODO: Should we run the whole experiment at once? By that I mean, do we
train all generative and predictive models at once rather before and use
the ones that did well?
TODO: Different verbosity levels?
TODO: Determine which is the best option to subset the dataset by. In
particular, which combination of the oracle/gp pair results in sequences
that are good in oracle predictions, but are not lead astry in
uninteresting regions based on the GT predictions (i.e. the preductions
should stay relatively close to each other from the different
iterations).
TODO: Do we save the full results for all samples?

Ok, so it seems that the original paper uses random sequences (~1000)
from the upper mean of the bimodal distribution to solve the problem to
form the GP. Since we dont have that liberty, we try using all values
greater than 0 for our GT.
The main issue is with the oracle. Even with all the rewightnign
schemes, the it still predicts the same value for all sequences (meaning
that all the sequences are essentially considered the same in terms of
the fitness values).

1. full dataset (w/ and w/o sampling), the one without sampling will
   likely perform worse since we are giving each sample equal weight.
   The question is then, do we want to bias our algorithm whle using
   sampling in this fashion?
2. subset of dataset
    * Greater than median (51)
    * Less than median (519)
    * Greater than mean (285)
    * Less than mean (285)
    * Greater than 0 (3XX)

What have we learned?
- The biggest factor in getting accurate "predictions" from the oracles
  is the access to "high quality" data. Since the data given is
  unsifficent (since only very few examples of high fitness variants
  were given), what we can conclude is that the model is unsifficient to
  capture what mutations differentiate a high fitness variant from a low
  one. We need a better "oracle" that handles not knowing what makes
  "good" samples better.
- In fact, even the GP is affected by this. This is beacuse the values y
  determine what sequences are good and bad. So, if you just give it
  good sequences to train on, it will think all sequences are "decent".
"""

import copy
import glob
import time
import typing
import multiprocessing as mp

import numpy as np
import scipy.stats

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from profit.models.torch import SequenceGPR, SequenceOracle, SequenceVAE
from profit.utils.data_utils.tokenizers import AminoAcidTokenizer
from profit.utils.testing_utils import avg_oracle_preds
from profit.utils.training_utils.torch import losses as L

from examples.gb1.data import load_dataset


ts = time.strftime("%Y-%b-%d-%H:%M:%S", time.gmtime())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor


# Preprocess + load the dataset
dataset = load_dataset("lstm", "primary", labels="Fitness", num_data=-1,
                       filetype="mdb", as_numpy=False, vocab="aa20")
_dataset = dataset[:]["arr_0"].long()
_labels = dataset[:]["arr_1"].view(-1)
# # Remove samples below a certain threshold
# high_idx = torch.where(_labels > _labels.mean())
# _dataset = _dataset[high_idx]
# _labels = _labels[high_idx]

# Initialize VAE and load weights (only for the prior)
tokenizer = AminoAcidTokenizer("aa20")
vocab_size = tokenizer.vocab_size
seqlen = _dataset.size(1)

def load_vaes(seqlen: int, vocab_size: int) -> typing.Tuple[SequenceVAE, ...]:
    """Initialize VAE and load weights (only for the prior)."""
    vae = SequenceVAE(seqlen, vocab_size, hidden_size=50, latent_size=20)
    vae_0 = copy.deepcopy(vae)
    vae_0.load_state_dict(torch.load("bin/3gb1/vae/2020-Apr-17-15:03:21/E0009.pt"))
    return vae, vae_0

# Initialize and load weights (all oracles)
paths = sorted(glob.glob("bin/3gb1/oracle/*/E*"))
metadata = ["all", "all-weighted", "g-mean", "l-mean", "g-median", "l-median", "g-zero"]
oracle_paths = dict(zip(metadata, paths))
oracle_stump = SequenceOracle(seqlen, vocab_size, hidden_size=50, out_size=2)
all_oracles = {}
for desc, path in oracle_paths.items():
    oracle = copy.deepcopy(oracle_stump)
    oracle.load_state_dict(torch.load(path))
    all_oracles[desc] = oracle

# Initialize and load weights (GPR)
paths = sorted(glob.glob("bin/3gb1/gpr/*"))
metadata = ["all", "g-mean", "l-mean", "g-median", "l-median", "g-zero"]
gpr_paths = dict(zip(metadata, paths))
all_gps = {desc: torch.load(path) for desc, path in gpr_paths.items()}


def cbas(oracles: typing.List[SequenceOracle], gp: SequenceGPR, vae: SequenceVAE,
         vae_0: SequenceVAE, topk: int = 10, verbose: bool = False):
    """Full CbAS procedure.

    The procedure stops when the weights are below a certain threshold,
    i.e. a proxy for the model will not explore new "interesting"
    sequences because they will not likely return any decent oracle values.
    """
    # Check if the models are the same (aka check if the number of parameters
    # are the same for each layer). For now, we just check if the keys have the
    # same values. TODO: more accurate comparison, i.e. size check for each layer.
    for vae_key, vae_0_key in zip(vae.__dict__.keys(), vae_0.__dict__.keys()):
        if (not vae_key.startswith("_") and
                vae.__dict__[vae_key] != vae_0.__dict__[vae_0_key]):
            raise ValueError("Models do not match!")

    num_iters = 20      # num iterations, how long do we want to run this for?
    num_samples = 500   # num of new seqs to sample per iteration
    sample = True       # should we sample the probs computed

    # Variance noise (only if using one oracle)
    if not isinstance(oracles, list):
        oracles = [oracles]
    num_oracles = len(oracles)
    homoscedastic = num_oracles == 1
    homo_var = 0.1

    # Weight sampling
    quantile = 0.95     # percentile (oracle samples above this are used)
    threshold = 1e-6    # weights cutoff (samples below this weight are removed)

    # Bookkeeping
    save_train_seqs = False # save seqs from initial dataset within topk?
    tracker = {
        "stats": torch.zeros(num_iters, 7),
        "topk": {
            "seq": np.zeros((num_iters, topk), dtype=object),
            "y_gt": torch.zeros(num_iters, topk),
            "y_oracle": torch.zeros(num_iters, topk)
        }
    }
    y_star = -np.inf

    # Iteratively generate new samples and determine their fitness
    for t in range(num_iters):
        # Sample (plausible) sequences from the latent distribution z
        zt = torch.randn(num_samples, vae.latent_size)
        if t > 0:
            # Since we don't perform any activation on the outputs of the
            # generative model (this is due to how the cross_entropy loss is
            # computed in torch), we need to apply a softmax to get probs for
            # each vocab.
            vae.eval()
            with torch.no_grad():
                # Compute softmax (logits -> probs) across each amino acid
                Xt_p = F.softmax(vae.decode(zt), dim=-1)
            if sample:
                # Sample from the computed probs to make potential sequences
                # that are somewhat outside our learned representation.
                Xt_sampled = torch.zeros_like(Xt_p)
                for i in range(Xt_p.size(0)):
                    for j in range(Xt_p.size(1)):
                        p = Xt_p[i, j].numpy()
                        k = np.random.choice(range(len(p)), p=p)
                        Xt_sampled[i, j, k] = 1.
                Xt_aa = torch.argmax(Xt_sampled, dim=-1)
            else:
                # Select vocab w/ highest prob
                Xt_aa = torch.argmax(Xt_p, dim=-1)
        else:
            Xt_aa = _dataset

        # Reconvert back to onehot
        Xt = torch.zeros(*Xt_aa.size(), vocab_size)
        Xt.scatter_(2, torch.unsqueeze(Xt_aa, 2), 1)

        # Evaluate sampled points on oracle(s)
        yt, yt_var = avg_oracle_preds(oracles, Xt)
        if homoscedastic:
            yt_var = torch.ones_like(yt) * homo_var
        # Evaluate sampled points on ground truth (aka GP)
        if t == 0 and _labels is not None:
            yt_gt = _labels
        else:
            yt_gt = gp.predict(Xt_aa, return_std=False)

        # Recompute weights for each sample (aka sequence)
        if t > 0:
            # Select log values (output of the current generative model) at
            # indicies where certain vocab (aka amino acid) were sampled. NOTE:
            # We compute the log for numerical stability (prevents overflow).
            log_pxt = torch.sum(torch.log(Xt_p) * Xt, dim=(1, 2))

            # Similarly for the inital generative model (prior)
            vae_0.eval()
            with torch.no_grad():
                X0_logp = F.log_softmax(vae_0.decode(zt), dim=-1)
            log_px0 = torch.sum(X0_logp * Xt, dim=(1, 2))
            w1 = torch.exp(log_px0 - log_pxt)

            # Threshold by selecting all y greater than certain quantile
            y_star_1 = np.percentile(yt, quantile*100)
            if y_star_1 > y_star:
                y_star = y_star_1
            w2 = scipy.stats.norm.sf(y_star, loc=yt, scale=np.sqrt(yt_var))
            weights = w1 * torch.Tensor(w2)
        else:
            weights = torch.ones(yt.shape[0])

        # Update tracker (bookkeeping)
        # Retrieve the topk values from the previous iteration, so that we can
        # compare from the previous run.
        if t == 0:
            # Set the first iter values to defaults
            temp_tracker = {
                "seq": np.array([None] * topk),
                "y_gt": torch.Tensor([-np.inf] * topk),
                "y_oracle": torch.Tensor([-np.inf] * topk),
            }
        else:
            temp_tracker = {
                "seq": tracker["topk"]["seq"][t-1:t].flatten(),
                "y_gt": tracker["topk"]["y_gt"][t-1:t].view(-1),
                "y_oracle": tracker["topk"]["y_oracle"][t-1:t].view(-1),
            }

        if save_train_seqs:
            # With every iteration, we can replace all topk sequences. As such,
            # we can simply just find the topk from each yt, combine the results
            # with the previous iteration, and choose the new topk variants from
            # that combined set (which has cardinality of 2*topk)
            yt_topk_idx = torch.sort(yt).indices[-topk:]
            yt_topk = yt[yt_topk_idx]
            yt_gt_topk = yt_gt[yt_topk_idx].squeeze()
            oracle_topk_seq = ["".join(tokenizer.decode(Xaa))
                               for Xaa in Xt_aa[yt_topk_idx].numpy()]

            # Concat tensors together
            temp_tracker["seq"] = np.concatenate((temp_tracker["seq"], oracle_topk_seq))
            temp_tracker["y_gt"] = torch.cat([temp_tracker["y_gt"], yt_gt_topk])
            temp_tracker["y_oracle"] = torch.cat([temp_tracker["y_oracle"], yt_topk])
        else:
            # Save all sequences which are not in original dataset
            for (xt_aa, y, y_gt) in zip(Xt_aa, yt, yt_gt):
                seq = "".join(tokenizer.decode(xt_aa.numpy()))
                if (seq not in tracker["topk"]["seq"] and
                        not torch.any((xt_aa == _dataset).all(axis=-1))):
                    # Concatenate from previous iteration
                    temp_tracker["seq"] = np.concatenate((temp_tracker["seq"], [seq]))
                    temp_tracker["y_gt"] = torch.cat([temp_tracker["y_gt"], y_gt.view(-1)])
                    temp_tracker["y_oracle"] = torch.cat([temp_tracker["y_oracle"], y.view(-1)])

        # Store only the topk sequences
        topk_idx = torch.sort(temp_tracker["y_oracle"]).indices[-topk:]
        tracker["topk"]["seq"][t] = np.array(temp_tracker["seq"])[topk_idx.tolist()]
        tracker["topk"]["y_gt"][t] = temp_tracker["y_gt"][topk_idx]
        tracker["topk"]["y_oracle"][t] = temp_tracker["y_oracle"][topk_idx]
        tracker["stats"][t] = torch.Tensor([yt.max(), yt.mean(), yt.std(),
                                            yt_gt.max(), yt_gt.mean(),
                                            yt_gt.std(), yt_var.mean()])

        # For printing purposes only, TODO: potentially remove
        if verbose:
            top1_idx = torch.argmax(tracker["topk"]["y_oracle"][t])
            print(t, tracker["topk"]["seq"][t, top1_idx],
                  tracker["topk"]["y_gt"][t, top1_idx],
                  tracker["topk"]["y_oracle"][t, top1_idx])
            print(tracker["stats"][t])

        # Train VAE model
        if t == 0:
            vae.load_state_dict(vae_0.state_dict())
        else:
            # Select samples whose weights > cutoff threshold
            cutoff_idx = torch.where(weights >= threshold)
            if cutoff_idx[0].size(0) == 0:
                break
            Xt = Xt[cutoff_idx]
            yt = yt[cutoff_idx]
            weights = weights[cutoff_idx]

            # Create dataset
            data_loader = DataLoader(
                dataset=TensorDataset(Xt, weights),
                batch_size=32,
                num_workers=mp.cpu_count(),
                pin_memory=torch.cuda.is_available(),
            )

            # Initialize optimizer; NOTE: Restart or continue from previous run?
            # Important because of weight decay param of AdamW.
            optimizer = optim.AdamW(vae.parameters(), lr=1e-3)

            # Restart training with new samples
            # NOTE: Do we restart or continue from the previous run of the VAE?
            # This affects the weight of the KL-divergence term (aka step param)
            step = 0
            vae.train()
            for epoch in range(10):
                for it, batch in enumerate(data_loader):
                    onehot = batch[0].to(device)
                    target = torch.argmax(onehot, dim=-1) # seqs
                    batch_size = onehot.size(0)
                    batch_weights = batch[1].to(device).view(-1, 1)
                    # Forward pass
                    pred, mu, logvar, _ = vae(onehot)
                    # Loss calculation
                    nll_loss, kl_loss, kl_weight = L.elbo_loss(
                        pred, target, mu, logvar, anneal_function=None, step=step,
                        k=0.0025, x0=2500, reduction="none")
                    # Reweight nll_loss w/ sample weights
                    nll_loss = (nll_loss * batch_weights).sum()
                    loss = (nll_loss + kl_weight * kl_loss) / batch_size
                    # Compute gradients and update params/weights
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1

    return tracker

vae, vae_0 = load_vaes(seqlen, vocab_size)
tracker, stats = cbas(all_oracles["g-mean"], all_gps["g-mean"], vae=vae,
                      vae_0=vae_0, topk=10, verbose=1)
print(tracker)
print(stats)

# # Run across all oracles and gp and dump results for topk
# dump = {}
# topk = 10
# for oracle_metadata, oracle in all_oracles.items():
#     for gp_metadata, gp in all_gps.items():
#         save_str = f"oracle={oracle_metadata}__gp={gp_metadata}"
#         # Copy VAE/VAE_0 since original obj. will still hold info from prev run
#         vae, vae_0 = load_vaes(seqlen, vocab_size)
#         tracker, statistics = cbas(oracle, gp, vae=vae, vae_0=vae_0, topk=topk, verbose=True)
#         print(tracker, statistics)
#         # Convert ndarrays to list, so that we can save dumps properly
#         tracker["seq"] = tracker["seq"].tolist()
#         tracker["y_gt"] = tracker["y_gt"].numpy().tolist()
#         tracker["y_oracle"] = tracker["y_oracle"].numpy().tolist()
#         dump[save_str] = tracker

# import os
# import json
# save_dir = "dumps/3gb1/cbas/"
# os.makedirs(save_dir, exist_ok=True)
# with open(os.path.join(save_dir, f"k={topk}.json"), "w") as dump_file:
#     json.dump(dump, dump_file)
