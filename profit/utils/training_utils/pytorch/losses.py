"""Custom loss functions."""

import math
import typing

import torch
from torch.nn import functional as F


# Anneal KL-divergence term, see: https://arxiv.org/abs/1511.06349
def _kl_anneal_function(anneal_function: str, step: int, k: float = 0.0025,
                        x0: int = 2500) -> float:
    if anneal_function == "logistic":
        return float(1/(1+math.exp(-k*(step-x0))))
    elif anneal_function == "linear":
        return min(1, step/x0)


def elbo_loss(pred: torch.Tensor, target: torch.Tensor, mu: torch.Tensor,
              logvar: torch.Tensor, anneal_function: str, step: int, k: float,
              x0: int, reduction: str = "sum",
              ) -> typing.Tuple[torch.Tensor, torch.Tensor, float]:
    """Compute variance of evidence lower bound (ELBO) loss.

    NOTE: The pred values should be logits (raw values). That is, no
    softmax/log softmax should be applied to the outputs. This is
    because F.cross_entropy() applies a F.log_softmax() internally
    before computing the negative log likelihood using F.nll_loss().
    """
    # Reconstruction loss
    # pred=(N,s,b), target=(N,s), where N=batch_size, s=seqlen, b=vocab_size
    pred = pred.permute(0, 2, 1) # Must be (N,b,s) for F.cross_entropy
    nll_loss = F.cross_entropy(pred, target, reduction=reduction)

    # KL Divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_weight = _kl_anneal_function(anneal_function, step, k, x0)

    return nll_loss, kl_loss, kl_weight


def gaussian_nll_loss(pred: torch.Tensor, target: torch.Tensor,
                      reduction: str = "sum") -> torch.Tensor:
    """Compute negative log-likelihood (NLL) loss of a `N(\\mu,\\sigma^2)`
    gaussian distribution.

    Usually, for regression-based tasks, neural networks often output a
    single value, `\\mu(x)`, and the parameters `\\theta` are optimized
    to minimize the mean-squared error (MSE) loss. This is because, we
    want to give more weight to the larger differences (aka outliers).

    However, the MSE does not capture the uncertainty in the prediction.
    Instead, we use a network that outputs two values, corresponding to
    the mean `\\mu(x)` and variance `\\sigma^2(x) > 0`, respectively [1].
    By treating the observed value `y \\in \\mathcal{N}(\\mu, \\sigma^2)`
    as a sample from a Gaussian distribution with the predicted mean and
    variance, we can minimize the likelihood function `p(y|x, \\theta)`,
    which represents the loss across all parameters `\\theta`. It can be
    easily shown that minimizing the NLL of our data with respect to
    `\\theta` is equivalent to minimizing the MSE between the observed
    `y` and our prediction [2, 3]. That is, `\\argmin(NLL) = \\argmin(MLE)`.

    NOTE: To ensure the positivity constraint on the variance, we pass
    the input through a softplus function `\\log (1+\\exp{(\\cdot)})`
    and add a minimum variance of `10^-6` for numerical stability.

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

    References:
    -----------
    [1] D. A. Nix and A. S. Weigend. Estimating the mean and variance of
        the target probability distribution. In Neural Networks, 1994.
    [2] http://willwolf.io/2017/05/18/minimizing_the_negative_log_likelihood_in_english/
    [3] https://stats.stackexchange.com/q/311331
    """
    N = pred.size(0)
    if N != target.size(0):
        raise ValueError(f"Sizes do not match ({N} != {target.size(0)}).")
    mu = pred[:, 0]
    # To ensure positive variance, we apply a softplus and add 1e-6
    var = F.softplus(pred[:, 1]) + 1e-6
    logvar = torch.log(var)
    target = target.squeeze(1)
    if reduction == "mean":
        return 0.5 * torch.log(torch.tensor([math.tau])) + 0.5 * torch.mean(logvar) \
            + 0.5 * torch.mean(torch.square(target - mu) / var)
    return 0.5 * N * torch.log(torch.tensor([math.tau])) + 0.5 * torch.sum(logvar) \
        + torch.sum(torch.square(target - mu) / (2 * var))
