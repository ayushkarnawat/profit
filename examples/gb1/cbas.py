"""Weighted Optimization (CbAS).

This procedure aims to optimize the generative model in such a fashion
that it generates plausible sequences (similar to the ones from the
original dataset) from the search space. It uses those sequences to
predict thier fitness on the oracle(s) and GT.

Steps:
------
1. Train + save generative model (on whole dataset)
2. Train + save oracle(s) w/ early stopping (thru validation data)
3. Train + save SequenceGP (we denote this as GT)..not ideal but works
   as a proxy for actual predictions. This was tested on the validation
   set.
4. For `t` iterations
    - Gather `n` samples from the latent vector `z`
    - Evaluate preds on oracles (avg them if using multiple ones to
      reduce noise)
    - Gather/Evaluate GT predictions
    - Compute sample weights
    - Track progress, taking care to save the sequences, their oracle
      and predictions.
    - Continue training generative model (with new samples) so that it
      can generate new (and hopefully better) samples in the search
      space that are "worth exploring".

Notes:
------
To train the generative model properly, we use 5000 randomly sampled
variants. This allows us to generate new "plausible" variants for each
timestep. We found that by using only the provided variants, the
generative model did not learn a good latent representation of the
sequences.

We save all the samples (sequence, oracle values, and gt values) for
each iteration. Although this leads to huge dumps (especially when there
are many sampled points per iteration), one benefit of this approach is
that this allows for more flexible stats. In particular, we can (a)
compute stats for each iteration, (b) retrieve only the topk across all
iterations by flattening and sorting the array, and (c) plot the
`q^{th}` percentile for each iteration to carefully observe how the
oracle is exploring the right regions of the search space.

In the original paper, the authors use randomly sampled sequences
(~1000) from the lower 20th percentile of the upper half of the bimodal
distribution (of the original dataset) to train the GP on. Since we
don't have that liberty, we try choosing a GT model that represents the
original dataset "well".

To select our optimal oracle and GT models, we train our model on the
following subsets of the given dataset:

1. All                      570
2. All (w/ sample weights)  570
3. Less than mean           519
4. Greater than 0           456
5. Greater than median      285
6. Less than median         285
7. Greater than mean        51

The question is then, do we want to bias our algorithm by using sample
weights in this fashion? It places more importance by statifying
samples, aka samples from a bin/quantile with a low count are given more
importance in the training loss. This "pushes" the model to predict
values that match those samples more than samples from a bin/quantile
with a high count.

What have we learned?
- The biggest factor in getting accurate "predictions" from the oracles
  is the access to "high quality" data. Since the data given is
  insufficent (only small number of high fitness variants were given),
  the resulting model predicts the similar values (with some small
  variance) no matter the sequence given. As such, most sequences are
  considered the same in terms of fitness values. This observation has
  led us to conclude is that the model does not have either enough (a)
  data or (b) features to capture what mutations differentiate a high
  fitness variant from a low one. We need a better model (aka "oracle")
  that is able to capture what makes a variant "good" versus "bad".
- The GP also sufferes from this, but to a lesser extent. This is
  because the fitness values determine what sequences are good and bad.
  So, if you just give it good sequences to train on, it will think all
  sequences are "decent", even though they might not be.


TODO: Should we run the whole experiment at once? By that we mean, do we
train all the generative and predictive models before and use
the ones that did well or do it all in one go?
TODO: Different verbosity levels?
TODO: Is it better to have more iterations, but less samples per
iteration? We speculate that more iterations will leads the VAE to
sample more effectively, since it has to try less samples per iteration.
But, on the other hand, having more samples might help explore regions
of sequences which might not have otherwise been explored.
"""

import os
import copy
import glob
import json
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


def cbas(oracles: typing.List[SequenceOracle],
         gp: SequenceGPR,
         vae: SequenceVAE,
         vae_0: SequenceVAE,
         num_iters: int = 50,
         num_samples: int = 200,
         sample: bool = True,
         homo_var: typing.Union[int, float] = 0.1,
         quantile: typing.Union[int, float] = 0.95,
         threshold: float = 1e-6,
         save_train_seqs: bool = False,
         lr: float = 1e-3,
         num_epochs: int = 10,
         verbose: bool = False) -> typing.Dict[str, typing.Any]:
    """Condition by Adaptive Sampling (CbAS) procedure.

    The procedure stops when the weights are below a certain threshold,
    which is a proxy that signifies that the model cannot find any more
    "interesting" sequences to sample (as they will likely not return
    any decent oracle values).

    Params:
    -------
    oracles: list of nn.Module
        Oracle model(s).

    gp: SequenceGPR
        Gaussian process regressor (denoted as the GT model).

    vae: nn.Module
        Generative model. Used to update/train the model on the newly
        sampled points :math:`y_t`.

    vae_0: nn.Module
        Generative (prior) model. Used to re-weight the samples
        :math`y_0` against current points :math:`y_t`.

    num_iters: int, default=50
        Number of iterations (upper limit) to run the CbAS algo for.

    num_samples: int, default=200
        Number of samples (aka sequences) to sample per iteration.

    sample: bool, default=True
        Sample the probabilities computed. If True, selects the amino
        acid (per position) based off the computed probs (aka weighted
        random choice). If False, selects the amino acid (per position)
        with the highest probabilty.

    homo_var: float, default=0.1
        Imputed variance for :math:`y_t`. Only used if there is one
        oracle.

    quantile: float, default=0.95
        Oracle values quantile. Chooses all :math:`y_t` greater than
        specified quantile.

    threshold: float, default=1e-6
        Sample weight cutoff. Samples below this threshold are removed
        from the training dataset.

    save_train_seqs: bool, default=False
        Save initial training dataset in tracker.

    lr: float, default=1e-3
        Optimizer learning rate.

    num_epochs: int, default=10
        Number of epochs to train the generative model on the new
        samples.

    verbose: bool, default=False
        Verbosity level.
    """
    # Check if the models are the same (aka check if the number of parameters
    # are the same per-layer). For now, we just check if the keys have the same
    # values. TODO: more accurate comparison, i.e. check size for each layer.
    for vae_key, vae_0_key in zip(vae.__dict__.keys(), vae_0.__dict__.keys()):
        if (not vae_key.startswith("_") and
                vae.__dict__[vae_key] != vae_0.__dict__[vae_0_key]):
            raise ValueError("Models do not match!")

    # Variance noise (only if using one oracle)
    if not isinstance(oracles, list):
        oracles = [oracles]
    homoscedastic = len(oracles) == 1
    if homoscedastic:
        if not isinstance(homo_var, (int, float)):
            raise TypeError(f"Invalid type `{type(homo_var)}` for homo_var. "
                            f"Should be float or int.")

    # Bookkeeping
    tracker = {
        "seq": np.zeros((num_iters, num_samples), dtype=object),
        "y_gt": torch.zeros(num_iters, num_samples),
        "y_oracle": torch.zeros(num_iters, num_samples),
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
        temp_tracker = {
            "seq": np.array([None] * num_samples),
            "y_gt": torch.Tensor([-np.inf] * num_samples),
            "y_oracle": torch.Tensor([-np.inf] * num_samples),
        }

        if save_train_seqs:
            # Since we only save atmost num samples per iteration, if the number
            # of values in the original dataset is greater than num samples, we
            # save only the top num samples. Alternatively, if it is less, we
            # save the all the values and use -np.inf (or None) for the rest.
            yt_samples_idx = torch.sort(yt).indices[-num_samples:]
            yt_samples = yt[yt_samples_idx]
            yt_gt_samples = yt_gt[yt_samples_idx].squeeze()
            oracle_seq = ["".join(tokenizer.decode(Xaa))
                          for Xaa in Xt_aa[yt_samples_idx].numpy()]

            temp_tracker["seq"] = np.concatenate((temp_tracker["seq"], oracle_seq))
            temp_tracker["y_gt"] = torch.cat([temp_tracker["y_gt"], yt_samples])
            temp_tracker["y_oracle"] = torch.cat([temp_tracker["y_oracle"], yt_gt_samples])
        else:
            # Save all sequences which are not in original dataset
            for (xt_aa, y, y_gt) in zip(Xt_aa, yt, yt_gt):
                seq = "".join(tokenizer.decode(xt_aa.numpy()))
                if not torch.any((xt_aa == _dataset).all(axis=-1)):
                    # Concatenate all valid sequences
                    temp_tracker["seq"] = np.concatenate((temp_tracker["seq"], [seq]))
                    temp_tracker["y_gt"] = torch.cat([temp_tracker["y_gt"], y_gt.view(-1)])
                    temp_tracker["y_oracle"] = torch.cat([temp_tracker["y_oracle"], y.view(-1)])

        # Store only the num_samples sequences
        samples_idx = torch.sort(temp_tracker["y_oracle"]).indices[-num_samples:]
        tracker["seq"][t] = temp_tracker["seq"][samples_idx.tolist()]
        tracker["y_gt"][t] = temp_tracker["y_gt"][samples_idx]
        tracker["y_oracle"][t] = temp_tracker["y_oracle"][samples_idx]

        # Print top sequence from current iteration
        if verbose:
            top1_idx = torch.argmax(tracker["y_oracle"][t])
            print(t, tracker["seq"][t, top1_idx], tracker["y_gt"][t, top1_idx],
                  tracker["y_oracle"][t, top1_idx])

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
            optimizer = optim.AdamW(vae.parameters(), lr=lr)

            # Restart training with new samples
            # NOTE: Do we restart or continue from the previous run of the VAE?
            # This affects the weight of the KL-divergence term (aka step param)
            step = 0
            vae.train()
            for epoch in range(num_epochs):
                for it, batch in enumerate(data_loader):
                    onehot = batch[0].to(device)
                    target = torch.argmax(onehot, dim=-1) # seqs
                    batch_size = onehot.size(0)
                    batch_weights = batch[1].to(device).view(-1, 1)
                    # Forward pass
                    pred, mu, logvar, _ = vae(onehot)
                    # Loss calculation
                    nll_loss, kl_loss, kl_weight = L.elbo_loss(
                        pred, target, mu, logvar, anneal_function=None,
                        step=step, k=0.0025, x0=2500, reduction="none")
                    # Reweight nll_loss w/ sample weights
                    nll_loss = (nll_loss * batch_weights).sum()
                    loss = (nll_loss + kl_weight * kl_loss) / batch_size
                    # Compute gradients and update params/weights
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1

    return tracker


save_dir = "dumps/3gb1/cbas/"
os.makedirs(save_dir, exist_ok=True)

# # Run across all oracles, gps and dump results for each "run"
# num_iters, num_samples = 50, 200
# num_total = num_iters * num_samples
# for oracle_type, oracle in all_oracles.items():
#     for gp_type, gp in all_gps.items():
#         # Reload VAEs since original objs. will still hold info from prev. run
#         vae, vae_0 = load_vaes(seqlen, vocab_size)
#         tracker = cbas(oracle, gp, vae=vae, vae_0=vae_0, num_iters=num_iters,
#                        num_samples=num_samples, verbose=1)
#         print(f"oracle={oracle_type}, gp={gp_type}", tracker)
#         # Convert ndarrays to list, so that we can save dumps properly
#         tracker["seq"] = tracker["seq"].tolist()
#         tracker["y_gt"] = tracker["y_gt"].numpy().tolist()
#         tracker["y_oracle"] = tracker["y_oracle"].numpy().tolist()
#         # Save dump of run to file
#         filepath = f"oracle={oracle_type}__gp={gp_type}__total={num_total}.json"
#         with open(os.path.join(save_dir, filepath), "w") as dump_file:
#             json.dump(tracker, dump_file)


# Use dataset (values greater than mean) trained on oracle and gp
oracle = gp = "g-mean"
n_iters = 50
n_samples = 200
total = n_iters * n_samples

# Load VAEs
vae, vae_0 = load_vaes(seqlen, vocab_size)
tracker = cbas(all_oracles["g-mean"], all_gps["g-mean"], vae=vae, vae_0=vae_0,
               num_iters=n_iters, num_samples=n_samples, verbose=1)
# Save dump to file
tracker["seq"] = tracker["seq"].tolist()
tracker["y_gt"] = tracker["y_gt"].numpy().tolist()
tracker["y_oracle"] = tracker["y_oracle"].numpy().tolist()
filepath = f"oracle={oracle}__gp={gp}__total={total}.json"
with open(os.path.join(save_dir, filepath), "w") as dump_file:
    json.dump(tracker, dump_file)
