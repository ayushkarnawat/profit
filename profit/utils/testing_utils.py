"""Testing utils."""

import typing

import torch
from torch.nn import functional as F


def avg_oracle_preds(oracles: typing.List[torch.nn.Module], X: torch.Tensor
                     ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """Average oracle predictions equally.

    NOTE: All oracles/models MUST have the same input and output shapes.
    """
    if not isinstance(oracles, list):
        oracles = [oracles]

    num_oracles = len(oracles)
    num_samples = X.size(0)
    means = torch.zeros(num_oracles, num_samples)
    var = torch.zeros(num_oracles, num_samples)
    for idx, oracle in enumerate(oracles):
        oracle.eval()
        with torch.no_grad():
            pred = oracle(X)
            means[idx, :] = pred[:, 0]
            var[idx, :] = F.softplus(pred[:, 1]) + 1e-6 # positivity constraint
    # Average preds across all oracles
    mu_star = torch.mean(means, dim=0)
    var_star = (1/num_oracles) * (torch.sum(var, dim=0) \
        + torch.sum(means**2, dim=0)) - mu_star**2
    return mu_star, var_star
