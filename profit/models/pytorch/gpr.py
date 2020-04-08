import typing
import warnings

import numpy as np
import torch

import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.lazy import DiagLazyTensor, SumLazyTensor
from gpytorch.models import ExactGP

from profit.utils.data_utils.substitution_matrices import BLOSUM62


class SubstitutionMatrixKernel(Kernel):
    r"""
    Computes the covariance matrix based on a variant of the amino acid
    SUbstitution Matrix (SUM) between inputs :math:`\mathbf{x}` and
    :math:`\mathbf{y}` [1]:

    .. math::

       \begin{equation*}
          k_{\text{BLOSUM}}(\mathbf{x}, \mathbf{y}) = \exp \left[
          \gamma \frac{\prod_{i,j=1}^{N} K(\mathbf{x}_i,\mathbf{y}_j)^{\beta}}
          {\sqrt{\prod_{i=1}^{N} K(\mathbf{x}_i, \mathbf{x}_i)^{\beta}
                 \prod_{j=1}^{N} K(\mathbf{y}_j, \mathbf{y}_j)^{\beta}}
          } \right]
       \end{equation*}

    where :math:`\mathbf{x}` and :math:`\mathbf{y}` are amino acid
    strings of the same length :math:`N`, :math:`\mathbf{x} =
    (\mathbf{x}_1, \dots, \mathbf{x}_N)`, :math:`\mathbf{y} =
    (\mathbf{y}_1, \dots, \mathbf{y}_N)`; :math:`\mathbf{x}, \mathbf{y}`
    are :math:`N`-mers. Hence, each entry in :math:`k_{\text{BLOSUM}}`
    represents the normalized form of the set of all :math:`N`-mers [2].

    Params:
    -------
    matrix: np.ndarray or torch.Tensor, default=None
        The amino acid substitution matrix. Denotes the substitution
        probability between each of the 20 natural amino acids. If None,
        uses BLOSUM62 matrix.

    alpha: float, default=0.1
        Imputed noise on diagonal.

    beta: float, default=0.1
        Exponentiation factor. In general, this should be small to avoid
        overflow errors when computing the product of the tensor.

    gamma: float, default=1.0
        Scaling factor (currently unlearnable).

    References:
    -----------
    [1] Shen et al. Introduction to the Peptide Binding Problem of Com-
        putational Immunology: New Results. Foundations of Computational
        Mathematics, 14(5):951–984, Oct 2014. ISSN 1615-3375. DOI:
        10.1007/s10208-013-9173-9.

    [2] Shawe-Taylor, John, and Nello Cristianini. Kernel methods for
        pattern analysis. Cambridge university press, 2004.
    """

    def __init__(self,
                 matrix: typing.Union[np.ndarray, torch.Tensor, None] = None,
                 alpha: float = 0.1,
                 beta: float = 0.1,
                 gamma: float = 1) -> None:
        super(SubstitutionMatrixKernel, self).__init__()
        # Map to torch tensor
        if matrix is None:
            matrix = BLOSUM62
        if not torch.is_tensor(matrix):
            matrix = torch.as_tensor(matrix)
        self.matrix = matrix
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False,
                last_dim_is_batch: bool = False) -> torch.Tensor:
        # Check if each value is 0 <= x1[i] < len(matrix).
        # NOTE: The value contained in x1, x2 represent indicies of the vocab
        # (i.e. amino acids). Each index in the BLOSUM62 matrix denotes the
        # substitution probabilty between the amino acids.
        if not torch.all(x1 < self.matrix.size(0)):
            raise ValueError("Check input x1 - all value should be [0, "
                             f"{self.matrix.shape[0]}).")
        if not torch.all(x2 < self.matrix.size(0)):
            raise ValueError("Check input x2 - all value should be [0, "
                             f"{self.matrix.shape[0]}).")

        # Check dims. If greater than 2 dims, return error; if less add dim
        if x1.ndim < 2:
            warnings.warn("x1.ndim < 2. Expanding across first dim.")
            x1 = x1.unsqueeze(0)
        if x2.ndim < 2:
            warnings.warn("x2.ndim < 2. Expanding across first dim.")
            x2 = x2.unsqueeze(0)

        # Repeat x1 across dim=1; x2 along dim=0 so that we can compute
        # covariance for all X_{ij} properly
        x1 = x1.unsqueeze(0).transpose(0,1).repeat(1, x1.size(0), 1)
        x2 = x2.unsqueeze(0).repeat(x2.size(0), 1, 1)

        # Retrieve substitution prob (between AAs at same position
        # between 2 sequences) across all amino acids.
        kij = torch.prod(self.matrix[[x1, x2]]**self.beta, dim=-1)
        kii = torch.prod(self.matrix[[x1, x1]]**self.beta, dim=-1)
        kjj = torch.prod(self.matrix[[x2, x2]]**self.beta, dim=-1)
        # Normalize kernel
        k = kij / (torch.sqrt(kii*kjj))
        noise = DiagLazyTensor(torch.Tensor([self.alpha]*k.size(0)))
        out = SumLazyTensor(torch.exp(self.gamma*k), noise)
        return out.diag() if diag else out


class SequenceGPR(ExactGP):
    r"""Gaussian process regression (GPR) for amino acid sequences.

    The (default) covariance kernel is some variant of the BLOSUM amino
    acids substitution matrix, which denotes the substitution probabilty
    between each of the 20 natural amino acids. This forms the prior.
    After it is fit to some training data, the hyperparameters of the
    kernel are optimized (based off the MLL). This optimized kernel
    allows us to predict the mean and variance of the testing points.

    Params:
    -------
    train_x: torch.Tensor
        Training features :math:`\mathbf{X}`.

    train_y: torch.Tensor
        Training targets :math:`\mathbf{y}`.

    likelihood: gpytorch.likelihoods.GaussianLikelihood
        The likelihood that defines the observational distribution.
        Since we're using exact inference, the likelihood must be
        Gaussian.
    """

    def __init__(self,
                 train_x: torch.Tensor,
                 train_y: torch.Tensor,
                 likelihood: gpytorch.likelihoods.GaussianLikelihood) -> None:
        super(SequenceGPR, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = SubstitutionMatrixKernel(matrix=BLOSUM62)

    def forward(self,
                x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
