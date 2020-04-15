"""Gaussian processes regressor(s)."""

import typing
import warnings
import numpy as np

from profit.utils.data_utils.substitution_matrices import BLOSUM62


class SequenceGPR:
    r"""Gaussian process regression (GPR) for amino acid sequences.

    The covariance matrix is computed by applying a variant of the amino
    acid SUbstitution Matrix (SUM) between inputs :math:`\mathbf{x}` and
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

    This covariance forms the prior and allows us to predict the mean
    and variance on the test set.

    Params:
    -------
    smatrix: np.ndarray, default=None
        The amino acid substitution matrix. Denotes the substitution
        probability between each of the 20 natural amino acids. If None,
        uses BLOSUM62 matrix.

    alpha: float, default=0.1
        Imputed noise on diagonal.

    beta: float, default=0.1
        Exponentiation factor. In general, this should be small to avoid
        overflow errors when computing the product of the tensor.

    gamma: float, default=1.0
        Scaling factor.

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
                 smatrix: typing.Optional[np.ndarray] = None,
                 alpha: float = 0.1,
                 beta: float = 0.1,
                 gamma: float = 1.0) -> None:
        if smatrix is None:
            smatrix = BLOSUM62
        self.smatrix = smatrix
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Computed during training
        self.x_train = None
        self.y_train = None
        self.k_ = None
        self.kinv_ = None


    # def _kernel(self, Xi, Xj):
    #     # Check if all values in Xi, Xj are <= len(smatrix)
    #     assert all(i < len(self.smatrix) for i in Xi)
    #     assert all(i < len(self.smatrix) for i in Xj)

    #     # Retrieve the substitution prob/covariance between the AAs based off
    #     # the vocab index.
    #     kij = np.prod(self.smatrix[[Xi, Xj]]**self.beta)
    #     kii = np.prod(self.smatrix[[Xi, Xi]]**self.beta)
    #     kjj = np.prod(self.smatrix[[Xj, Xj]]**self.beta)
    #     # Normalize the kernel, following Shawe-Taylor J., Cristianini N. (2004)
    #     # Kernel Methods for Pattern Analysis. Cambridge University Press.
    #     k = kij / (np.sqrt(kii*kjj))
    #     return np.exp(self.gamma*k)


    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        r"""Compute the covariance kernel.

        Params:
        -------
        x_train: np.ndarray
            Training features :math:`\mathbf{X}`.

        y_train: np.ndarray
            Training targets :math:`\mathbf{y}`.
        """
        # Convert values to ints, if necessary
        if x_train.dtype.kind != "i":
            warnings.warn(f"Values in x_train of type `{str(x_train.dtype)}`. "
                          "Converting to ints.")
            x_train = x_train.astype(np.int)

        # Check if each value is 0 <= x_train[i] < len(smatrix). This is because
        # x_train represents indicies of the vocab (i.e. amino acids).
        if not np.all(x_train >= 0) and np.all(x_train < self.smatrix.shape[0]):
            raise ValueError("Check input x_train - all value should be [0, "
                             f"{self.smatrix.shape[0]}).")

        # Save params for prediction
        self.x_train = x_train
        self.y_train = y_train

        # Repeat x1 along dim=1 (similarly x2 along dim=0) so that we can
        # compute covariance between all permutations of x_train
        N = x_train.shape[0]
        xi = np.tile(np.transpose(np.expand_dims(x_train, axis=0),
                                  axes=(1, 0, 2)), reps=(1, N, 1))
        xj = np.tile(np.expand_dims(x_train, axis=0), reps=(N, 1, 1))

        # Compute covariance kernel k(x_i, x_j)
        # Retrieve substitution prob (between AAs at same position
        # between 2 sequences) across all amino acids.
        kij = np.prod(self.smatrix[(xi, xj)]**self.beta, axis=-1)
        kii = np.prod(self.smatrix[(xi, xi)]**self.beta, axis=-1)
        kjj = np.prod(self.smatrix[(xj, xj)]**self.beta, axis=-1)
        k = kij / (np.sqrt(kii*kjj))            # normalize kernel
        noise = self.alpha * np.eye(k.shape[0]) # noise along diag
        self.k_ = np.exp(self.gamma*k) + noise

        # Invert covariance kernel k; needed for prediction
        self.kinv_ = np.linalg.inv(self.k_)


    def predict(self, x_star: np.ndarray,
                return_std: bool = False) -> typing.Tuple[np.ndarray, ...]:
        r"""Compute the mean and variance of the given points using the
        computed covariance kernel.

        NOTE: The last dim (which describes the protein's length) of the
        test points must be the same as the last dim of the training
        points.

        Params:
        -------
        x_star: np.ndarray
            Testing features :math:`\mathbf{X^*}`.

        return_std: bool, default=False
            If True, returns the std (uncertainty) of the prediction.
        """
        # Check if the sequence lengths match
        if x_star.shape[-1] != self.x_train.shape[-1]:
            raise ValueError(f"Sequence lengths do not match ({x_star.shape[-1]}"
                             f"!= {self.x_train.shape[-1]}).")

        # Convert values to ints, if necessary
        if x_star.dtype.kind != "i":
            warnings.warn(f"Values in x_star of type `{str(x_star.dtype)}`. "
                          "Converting to ints.")
            x_star = x_star.astype(np.int)

        # Check if each value is 0 <= x_star[i] < len(smatrix). This is because
        # x_star represents indicies of the vocab (i.e. amino acids).
        if not np.all(x_star >= 0) and np.all(x_star < self.smatrix.shape[0]):
            raise ValueError("Check input x_star - all value should be [0, "
                             f"{self.smatrix.shape[0]}).")

        # Repeat x1 along dim=1 (similarly x2 along dim=0) so that we can
        # compute covariance between x_star and x_train
        M = x_star.shape[0]
        N = self.x_train.shape[0]
        x_star = np.tile(np.transpose(np.expand_dims(
            x_star, axis=0), axes=(1, 0, 2)), reps=(1, N, 1))
        x_train = np.tile(np.expand_dims(self.x_train, axis=0), reps=(M, 1, 1))

        # Compute kernel k(x_star, x_train)
        kij = np.prod(self.smatrix[(x_star, x_train)]**self.beta, axis=-1)
        kii = np.prod(self.smatrix[(x_star, x_star)]**self.beta, axis=-1)
        kjj = np.prod(self.smatrix[(x_train, x_train)]**self.beta, axis=-1)
        k = kij / (np.sqrt(kii*kjj)) # normalize kernel
        k_star = np.exp(self.gamma*k)

        # TODO: Confirm these are equivalent
        # M = x_star.shape[0]
        # N = len(self.k_)
        # total = M * N
        # k_star = np.zeros((M, N))
        # for i in range(M):
        #     for j in range(N):
        #         kij = self._kernel(x_star[i], self.x_train[j])
        #         k_star[i, j] = kij
        y_mean = np.matmul(k_star, np.matmul(self.kinv_, self.y_train))
        if return_std:
            # Since K(x*,x*)=1, the variance will be e^(gamma*1.)
            y_var = np.repeat(np.exp(self.gamma*1.), M)
            y_var -= np.einsum("ij,ij->i",
                               np.dot(k_star, self.kinv_), k_star)
            # Check if any of the variances is negative because of
            # numerical issues. If yes, set variance to 0.
            y_var_negative = y_var < 0
            if np.any(y_var_negative):
                warnings.warn("Predicted variances smaller than 0. "
                              "Setting those variances to 0.")
                y_var[y_var_negative] = 0.0
            return y_mean, np.sqrt(y_var)
        return y_mean
