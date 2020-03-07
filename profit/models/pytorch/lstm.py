"""LSTM (long short-term memory) RNN model."""

from typing import Optional, Tuple, Union

import torch
from torch import nn, Tensor
from torch.nn import init
from torch.nn import functional as F


class LSTMLayer(nn.Module):
    """LSTM module/layer.

    NOTE: We are not using the default `nn.LSTM` module with `dropout`
    param specified as that requires `num_layers > 1` since the dropout
    is applied at the end of each LSTM layer (expect for the last layer).
    Futhermore, we also want to define different `input_size` and
    `hidden_size` for each individual LSTM layer. This is not possible
    with the default `nn.LSTM` module. Hence, we have to apply the dropout
    before the LSTM. This will allow us to define the full encoder with
    diff input dims (i.e. LSTM -> dropout -> LSTM -> dropout -> ... -> LSTM).

    Params:
    -------
    input_size: int, default=128
        Number of expected features in the input `x`.

    hidden_size: int, default=128
        Number of features in the hidden state `h`.

    dropout: float, default=0.
        Percentage of input/hidden units to drop (aka set weight 0)
        during each training step.

    Inputs:
    -------
    input: torch.Tensor, shape `(seq_len, batch, input_size)`
        Tensor containing the features of the input sequence.

    (h_0, c_0): tuple of torch.Tensors, shape `(num_layers * num_directions, batch, hidden_size)`
        Tensors containing initial hidden/cell state for each element
        in the batch, respectively. If not provided, both **h_0** and
        **c_0** default to zero. If the LSTM is bidirectional,
        num_directions should be 2, else it should be 1.

    Outputs:
    --------
    output: torch.Tensor, shape `(batch, seqlen, num_directions * hidden_size)`
        Tensor containing the output features `(h_t)` from the last
        layer of the LSTM, for each `t`.

    (h_n, c_n): tuple of torch.Tensors, shape `(num_layers * num_directions, batch, hidden_size)`
        Tensor containing the hidden and cell state for `t = seq_len`.
    """

    def __init__(self,
                 input_size: int = 128,
                 hidden_size: int = 1024,
                 dropout: float = 0.) -> None:
        super(LSTMLayer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Computation performed at every call."""
        inputs = self.dropout(inputs)
        self.lstm.flatten_parameters()
        return self.lstm(inputs)


class LSTMPooler(nn.Module):
    """Pooling layer for LSTM module/layer.

    We "pool" the model by taking the hidden state corresponding to the
    first token.

    Params:
    -------
    hidden_size: int, default=1024
        Number of features in the hidden state `h`.

    num_hidden_layers: int, default=3
        Number of hidden layers.

    Inputs:
    -------
    input: torch.Tensor, shape `(batch, hidden_size, 2*num_hidden_layers)`
        Hidden pooled embedding.

    Outputs:
    --------
    pooled_output: torch.Tensor, shape `(batch, hidden_size)`
        Pooled represetation of input.
    """

    def __init__(self,
                 hidden_size: int = 1024,
                 num_hidden_layers: int = 3) -> None:
        super(LSTMPooler, self).__init__()
        self.scalar_reweighting = nn.Linear(2*num_hidden_layers, 1)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, inputs: Tensor) -> Tensor:
        """Computation performed at every call."""
        # NOTE: The input tensor must be 3D, so that after the dense layer
        # is applied (aka scalar reweightng), it can remove the 3rd dim (which
        # is now of size 1).
        pooled_output = self.scalar_reweighting(inputs).squeeze(2)
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class LSTMEncoder(nn.Module):
    """Encoding layer for LSTM mdoels.

    Params:
    -------
    input_size: int, default=128
        Number of expected features in the input `x`.

    hidden_size: int, default=1024
        Number of features in the hidden state `h`.

    num_hidden_layers: int, default=3
        Number of hidden layers.

    hidden_dropout: float, default=0.1
        Percentage of input/hidden units to drop (aka set weight 0)
        during each training step.

    return_hs: bool, default=False
        If True, returns the hidden states in the output. If False, the
        hidden states will not be returned.
    """

    def __init__(self,
                 input_size: int = 128,
                 hidden_size: int = 1024,
                 num_hidden_layers: int = 3,
                 hidden_dropout: float = 0.1,
                 return_hs: bool = False) -> None:
        super(LSTMEncoder, self).__init__()
        forward_lstm = [LSTMLayer(input_size, hidden_size)]
        reverse_lstm = [LSTMLayer(input_size, hidden_size)]
        for _ in range(1, num_hidden_layers):
            forward_lstm.append(LSTMLayer(hidden_size, hidden_size, hidden_dropout))
            reverse_lstm.append(LSTMLayer(hidden_size, hidden_size, hidden_dropout))
        self.forward_lstm = nn.ModuleList(forward_lstm)
        self.reverse_lstm = nn.ModuleList(reverse_lstm)
        self.return_hs = return_hs

    def forward(self, inputs: Tensor,
                input_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tuple[Tensor, ...]]:
        """Computation performed at every call."""
        forward_output = inputs
        all_hidden_states = (inputs,)
        # Forward pass through LSTM layers
        all_forward_pooled = ()
        for layer in self.forward_lstm:
            forward_output, forward_pooled = layer(forward_output)
            all_forward_pooled += (forward_pooled[0],)
            all_hidden_states += (forward_output,)

        # Reverse pass thorough LSTM layers
        all_reverse_pooled = ()
        reverse_output = self._reverse_sequence(inputs, input_mask)
        for layer in self.reverse_lstm:
            reverse_output, reverse_pooled = layer(reverse_output)
            all_reverse_pooled += (reverse_pooled[0],)
            all_hidden_states += (reverse_output,)
        reverse_output = self._reverse_sequence(reverse_output, input_mask)

        # Combine both sequence and pooled representations
        output = torch.cat((forward_output, reverse_output), dim=2)
        pooled = all_forward_pooled + all_reverse_pooled
        pooled = torch.stack(pooled, dim=3).squeeze(0)
        outputs = (output, pooled)
        if self.return_hs:
            outputs += (all_hidden_states,)
        return outputs  # sequence_embedding, pooled_embedding, (hidden_states)

    @staticmethod
    def _reverse_sequence(sequence: Tensor,
                          input_mask: Optional[Tensor] = None) -> Tensor:
        """Reverse the sequence. Useful for reverse pass through LSTM."""
        if input_mask is None:
            idx = torch.arange(sequence.size(1) - 1, -1, -1)
            reversed_sequence = sequence.index_select(1, idx, device=sequence.device)
        else:
            sequence_lengths = input_mask.sum(1)
            reversed_sequence = []
            for seq, seqlen in zip(sequence, sequence_lengths):
                idx = torch.arange(seqlen - 1, -1, -1, device=seq.device, dtype=torch.int64)
                seq = seq.index_select(0, idx)
                seq = F.pad(seq, [0, 0, 0, sequence.size(1) - int(seqlen)])
                reversed_sequence.append(seq)
            reversed_sequence = torch.stack(reversed_sequence, 0)
        return reversed_sequence


class LSTMModel(nn.Module):
    """LSTM model for protein-related tasks.

    Params:
    -------
    vocab_size: int
        The size of the dictionary, aka number of unique tokens.

    input_size: int, default=128
        Number of expected features in the input `x`.

    hidden_size: int, default=1024
        Number of features in the hidden state `h`.

    num_hidden_layers: int, default=3
        Number of hidden layers.

    num_outputs: int, default=1
        Number of outputs, i.e. values to predict. Equal to the number
        of targets values we have labels for.

    hidden_dropout: float, default=0.1
        Percentage of input/hidden units to drop (aka set weight 0)
        during each training step.

    return_hs: bool, default=False
        If True, returns the hidden states in the output. If False, the
        hidden states will not be returned.
    """

    def __init__(self,
                 vocab_size: int,
                 input_size: int = 128,
                 hidden_size: int = 1024,
                 num_hidden_layers: int = 3,
                 num_outputs: int = 1,
                 hidden_dropout: float = 0.1,
                 return_hs: bool = False) -> None:
        super(LSTMModel, self).__init__()
        self.embed_matrix = nn.Embedding(vocab_size, input_size)
        self.encoder = LSTMEncoder(input_size, hidden_size, num_hidden_layers,
                                   hidden_dropout, return_hs)
        self.pooler = LSTMPooler(hidden_size, num_hidden_layers)
        self.out = nn.Linear(hidden_size, num_outputs)
        self.return_hs = return_hs
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize the weights."""

        def _init_weights(module: nn.Module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # TODO: Should this be xavier_uniform_?
                # TODO: Should we allow user to specify the mean/std?
                init.normal_(module.weight, mean=0., std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                init.zeros_(module.bias)

        # Initialize weights across all (sub)modules
        self.apply(_init_weights)

    def forward(self, inputs: Tensor,
                input_mask: Optional[Tensor] = None
                ) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, ...]]]:
        """Computation performed at every call."""
        if input_mask is None:
            input_mask = torch.ones_like(inputs)

        inputs = inputs.long() # nn.Embedding only works with LongTensors
        embedding_output = self.embed_matrix(inputs)
        encoded = self.encoder(embedding_output, input_mask=input_mask)
        pooled_outputs = self.pooler(encoded[1])

        # NOTE: The (general) output below is used to store the sequence and
        # pooled representations after a forward pass through the model.
        # sequence_output = encoded[0]
        # hidden_states = encoded[2:]
        # return (sequence_output, pooled_outputs, hidden_states)

        # NOTE: Since we are only concerned about the value prediction task, we
        # apply the dense layer ONLY on the pooled embeddings. Additionally,
        # since this is a regression task, we apply linear (aka no) activation.
        pred = self.out(pooled_outputs)
        return pred
        # prediction, (hidden_states)
        # return pred if not self.return_hs else pred, encoded[2:]
