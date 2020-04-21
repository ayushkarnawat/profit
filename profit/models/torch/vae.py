"""Variational autoencoder model."""

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class BaseVAE(nn.Module):
    """Base class for creating variational autoencoders (VAEs).

    The module is designed to connect user-specified encoder/decoder
    layers to form a latent space representation of the data.

    A general overview of the model can be described by:
    https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html
    """

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()


    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Builds the encoded representation of the input.

        The encoded model outputs the mean and logvar of the latent
        space embeddings/distribution, or in more mathematical terms,

        :math:: `q(z|x) = \\mathcal{N}(z| \\mu(x), \\sigma(x))`
        """
        raise NotImplementedError


    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparamaterization trick.

        Computes the latent vector (`z`), which is a compressed low-dim
        representation of the input.

        This trick allows us to express the gradient of the expectation
        as the expectation of the gradient [1]. Additionally, it makes
        the variance of the estimate an order of magnitude lower than
        without using it. This allows us to compute the gradient during
        the backward pass more accurately, with better estimates [2].

        References:
        -----------
        -[1] https://gregorygundersen.com/blog/2018/04/29/reparameterization/
        -[2] https://stats.stackexchange.com/a/226136
        """
        std = torch.exp(0.5*logvar)
        # eps=N(0,I), where the I is an identity matrix of same size as std
        eps = torch.randn_like(std)
        return mu + std*eps


    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes the sampled latent vector (`z`) into the reconstructed
        output (`x'`).

        Ideally, the reconstructed output (`x'`) is identical to the
        original input (`x`).
        """
        raise NotImplementedError


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


class SequenceVAE(BaseVAE):
    """CbAS VAE model for (one-hot) encoded sequences."""

    def __init__(self,
                 seqlen: int,
                 vocab_size: int,
                 hidden_size: int = 64,
                 latent_size: int = 20) -> None:
        super(SequenceVAE, self).__init__()
        self.seqlen = seqlen
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # Probablistic encoder
        self.fc1 = nn.Linear(seqlen * vocab_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)
        # Probablistic decoder
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, seqlen * vocab_size)
        # Reshape occurs here (see self.decode())
        # size is now: (seqlen * vocab_size) -> (seqlen, vocab_size)
        self.fc5 = nn.Linear(vocab_size, vocab_size)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Flatten (n, seqlen, vocab_size) -> (n, seqlen * vocab_size)
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Input tensor: Latent vector z = (num_samples, latent_size)
        h3 = F.relu(self.fc3(z))
        h4 = self.fc4(h3)
        reshaped = h4.view(h4.size(0), self.seqlen, self.vocab_size)
        # Return logits since F.cross_entropy computes log_softmax internally
        return self.fc5(reshaped)
