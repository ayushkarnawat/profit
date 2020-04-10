"""Testing utils.

TODO: Have more general prediction util where the user provides the mu
and var for each of the oracles as input rather than just passing in the
oracles.
"""

from typing import List

import torch
from torch.nn import functional as F


def avg_oracle_preds(oracles: List[torch.nn.Module], Xt: torch.Tensor):
    """Average oracle predictions equally.

    NOTE: All oracles/models MUST have the same input and output shapes.
    """
    if not isinstance(oracles, list):
        oracles = [oracles]

    num_oracles = len(oracles)
    num_samples = Xt.size(0)
    means = torch.zeros(num_oracles, num_samples)
    var = torch.zeros(num_oracles, num_samples)
    for idx, oracle in enumerate(oracles):
        oracle.eval()
        with torch.no_grad():
            pred = oracle(Xt)
            means[idx, :] = pred[:, 0]
            var[idx, :] = F.softplus(pred[:, 1]) + 1e-6 # positivity constraint
    # Average preds across all oracles
    mu_star = torch.mean(means, dim=0)
    var_star = (1/num_oracles) * (torch.sum(var, dim=0) \
        + torch.sum(means**2, dim=0)) - mu_star**2
    return mu_star, var_star
