"""Weighted Optimization (CbAS).

Steps:
------
NOTE: For all the training steps, just use the whole training dataset.
1. Train + save VAE (on whole dataset)
2. Train + save oracle(s) w/ early stopping (thru validation data)
3. Train + save SequenceGP (we denote this as GT)..not ideal but works
   as a proxy for actual predictions. This was tested on the validation
   set.
4. For t iterations
  - Gather samples from latent
      - Gather all possible samples
      - Remove samples that are not variants from the traning set?...
        this ensure we only select from the 160K samples)
  - Evaluate preds on oracles (avg them using balaji scheme if using
    multiple ones) and gather GT predictions
  - Compute weighting scheme
  - Track progress, in particular which is the best prediction. Also
    save, per iteration, the space (z) the values occupy and their
    prediction as a color value. Alternatively, one can make the x axis
    and y exis as the 4 mutated sequences, with the output of the
    optimization scheme returning the fitness.
  - Retrain VAE so that it can generate new (and hopefully better)
    samples. It should ideally be giving predictions where we want to
    sample next.

This procedure aims to optimize the VAE is such a fashion that it
generates plausible sequences (similar to the ones in the search space).
And uses them to predict on the oracles.

TODO: Should we run the whole experiment at once? By that I mean, do we
train all generative and predictive models at once rather before and use
the ones that did well? 
TODO: Train VAE model with validation set as well?
"""

import copy
import time
import multiprocessing as mp

import numpy as np
import scipy.stats

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from profit.models.pytorch.vae import SequenceVAE
from profit.models.pytorch.lstm import LSTMModel
from profit.utils.data_utils.tokenizers import AminoAcidTokenizer
from profit.utils.training_utils.pytorch import losses as L

from examples.gb1.data import load_dataset
from examples.gb1.seq_gp import SequenceGPR


ts = time.strftime("%Y-%b-%d-%H:%M:%S", time.gmtime())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

# FOR ORACLE MODEL(S)
num_oracles = 1
homoscedastic = num_oracles == 1
homo_y_var = 0.1

# For weights
quantile = 0.95
threshold = 1e-6

# Preprocess + load the dataset
dataset = load_dataset("lstm", "primary", labels="Fitness", num_data=-1,
                       filetype="mdb", as_numpy=False, vocab="aa20")
_dataset = dataset[:]["arr_0"].long()
_labels = dataset[:]["arr_1"].view(-1)

# Initialize VAE and load weights (only for the prior)
tokenizer = AminoAcidTokenizer("aa20")
vocab_size = tokenizer.vocab_size
seqlen = _dataset.size(1)
vae = SequenceVAE(seqlen, vocab_size, hdim=64, latent_size=20)
vae_0 = copy.deepcopy(vae)
vae_0.load_state_dict(torch.load("bin/3gb1/vae/2020-Apr-08-17:53:11/E0020.pt"))

# Initialize and load weights (oracle)
oracle = LSTMModel(vocab_size, input_size=64, hidden_size=128, num_layers=2,
                   num_outputs=2, hidden_dropout=0.25)
oracle.load_state_dict(torch.load("bin/3gb1/lstm/2020-Apr-08-17:36:05/E0002.pt"))

# Train GPR using BLOSUM62 substitution matrix
gp = SequenceGPR()
gp.fit(_dataset.numpy(), _labels.numpy())

# Bookkeeping
# traj = np.zeros((iters, 7))
# oracle_samples = np.zeros((iters, samples))
# gt_samples = np.zeros((iters, samples))
oracle_max_seq = None
oracle_max = -np.inf
gt_of_oracle_max = -np.inf
y_star = -np.inf

# Iteratively generate new samples and determine their fitness
for t in range(20):
    # Sample plausible seqs from the latent distribution
    zt = torch.randn(32, 20) # size=(samples=32, latent_size=20)
    if t > 0:
        # Since we don't perform any activation on the outputs of the generative
        # model initially (this is due to how the cross_entropy loss is computed
        # in torch), we need to apply a softmax to get probabilities for each
        # vocab.
        vae.eval()
        with torch.no_grad():
            # Compute softmax across amino acid in the sequence
            Xt_p = F.softmax(vae.decode(zt), dim=-1)
        Xt = torch.argmax(Xt_p, dim=-1) # convert from onehot -> seqs

        # NOTE: Alternatively, we could also sample from the computed probs to
        # make potential sequences that are outside our learned representation.
    else:
        Xt = _dataset

    # Evaluate sampled points on oracle(s) and ground truth (aka GP)
    # NOTE: If using multiple oracles, we should (a) average the predictions (mu
    # and var) and (b) NOT impute homoscedastic noise.
    oracle.eval()
    with torch.no_grad():
        pred = oracle(Xt)
    yt, yt_var = pred[:, 0], pred[:, 1]
    if homoscedastic:
        yt_var = torch.ones_like(yt) * homo_y_var
    # Determine ground truth values
    if t == 0 and _labels is not None:
        yt_gt = _labels
    else:
        yt_gt = torch.Tensor(gp.predict(Xt.numpy(), return_std=False))

    # Recompute weights for each sample (aka sequence)
    if t > 0:
        # Select log values (output of the current generative model) at indicies
        # where certain vocab (aka amino acid) were sampled. NOTE: We compute
        # the log for numerical stability (prevents overflow).
        batch_size, seqlen = Xt.size()
        Xt_onehot = torch.zeros(batch_size, seqlen, vocab_size)
        Xt_onehot.scatter_(2, torch.unsqueeze(Xt, 2), 1)
        log_pxt = torch.sum(torch.log(Xt_p) * Xt_onehot, dim=(1, 2))

        # Similarly for the inital generative model (prior)
        vae_0.eval()
        with torch.no_grad():
            X0_logp = F.log_softmax(vae_0.decode(zt), dim=-1)
        log_px0 = torch.sum(X0_logp * Xt_onehot, dim=(1, 2))
        w1 = torch.exp(log_px0 - log_pxt)

        # Threshold by selecting all y greater than certain quantile
        y_star_1 = np.percentile(yt, quantile*100)
        if y_star_1 > y_star:
            y_star = y_star_1
        w2 = scipy.stats.norm.sf(y_star, loc=yt, scale=np.sqrt(yt_var))
        weights = w1 * torch.Tensor(w2)
    else:
        weights = torch.ones(yt.shape[0])

    # Bookkeeping
    print(weights) # TODO: Remove
    # HMMMM, for some reason it is not getting good weights (and subsequently
    # good predictions from oracle for the new points sampled) after first iteration
    yt_max_idx = torch.argmax(yt)
    yt_max = yt[yt_max_idx]
    if yt_max > oracle_max:
        oracle_max = yt_max
        oracle_max_seq = "".join(tokenizer.decode(Xt[yt_max_idx].numpy()))
        gt_of_oracle_max = yt_gt[yt_max_idx]
        print(oracle_max_seq, oracle_max, gt_of_oracle_max)

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

        # Onehot encode seqs; need to redo since some Xt might have been removed
        batch_size, seqlen = Xt.size()
        Xt_onehot = torch.zeros(batch_size, seqlen, vocab_size)
        Xt_onehot.scatter_(2, torch.unsqueeze(Xt, 2), 1)

        # Create dataset
        data_loader = DataLoader(
            dataset=Xt_onehot,
            batch_size=32, # NOTE: In its current implementation, this has to be the same as weights. It can't be less
            num_workers=mp.cpu_count(),
            pin_memory=torch.cuda.is_available(),
        )

        # Initialize optimizer; TODO: Restart or continue from previous? This is
        # important because of weight decay.
        optimizer = optim.AdamW(vae.parameters(), lr=1e-3)

        # Restart training with new samples
        step = 0 # TODO: Restart or continue from previous??
        for epoch in range(10): # TODO: it_epochs=10, can be changed
            for it, batch in enumerate(data_loader):
                onehot = batch.to(device)
                target = torch.argmax(onehot, dim=-1) # seqs
                # Forward pass
                pred, mu, logvar, z = vae(onehot)
                # Loss calculation
                nll_loss, kl_loss, kl_weight = L.elbo_loss(pred, target, mu, \
                    logvar, anneal_function="logistic", step=step, k=0.0025, \
                    x0=2500, reduction="none")
                # Reweight nll_loss w/ sample weights
                # NOTE: Should we normalize sample weights: https://discuss.pytorch.org/t/25530/4?
                nll_loss = (nll_loss * weights.view(-1, 1)).sum()
                loss = (nll_loss + kl_weight * kl_loss) / batch_size
                # Compute gradients and update params/weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1


##### ACTION ITEMS
# TODO: Use multiple oracles? THis requires averaging the predictions y and
# ensuring positive variance thru softplus activation....
# TODO: Figure out reweighting scheme properly when retraining.... do we
# normalize the weights or keep them unchanged?
# TODO: Are the sample weights being calculated correctly?
# TODO: Try sampling the sequences rather than selecting those with the highest
# probability of being generated from the latent distribution. THIS could be the
# key as the model elarns to explore the space properly.... alternatively we
# could keep a track of all the sequences processed in previous iterations and
# select only those that have not been looked at previously...
# TODO: How is the latent space changing...

# TODO: Try using oracle that is given in CbAS (just densely
# connected)...although according to their paper this doesn't matter that much
# since they assume the oracle is not that good. 
# TODO: Try this whole method on CbAS data? Could the issue be with the actual
# dataset? It is quite possible...
