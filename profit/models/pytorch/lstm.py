"""LSTM (long short-term memory) RNN model."""

import torch.nn as nn


class LSTMLayer(nn.Module):
    """LSTM module/layer. 
    
    NOTE: Assumes the input tensor is always batch_first.

    TODO: Should we just use the original LSTM layer, and add a dropout 
    parameter to it? The only difference between this and what is in-
    built is that the dropout occurs before the LSTM layer instead of before.  

    Params:
    -------
    input_size: int
        The number of expected features in the input `x`.

    hidden_size: int
        The number of features in the hidden state `h`.

    dropout: float, default=0.
        Percentage of input/hidden units to drop (aka set weight 0) 
        during each training step.
    """

    def __init__(self, input_size: int=128, hidden_size: int=1024, dropout: float=0.):
        # Define all nodes in this module/layer
        super(LSTMLayer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs):
        """Forward pass through nodes (aka logic/computation performed 
        at every call).
        """
        inputs = self.dropout(inputs)
        self.lstm.flatten_parameters()
        return self.lstm(inputs)


class LSTMPooler(nn.Module):
    """Pooling layer for LSTM module/layer."""

    def __init__(self, hidden_size: int=1024, num_hidden_layers: int=3):
        # Define all nodes in this module/layer. 
        super(LSTMPooler, self).__init__()
        self.scalar_reweighting = nn.Linear(2*num_hidden_layers, 1)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh() # NOTE: Change to be more general?

    def forward(self, inputs):
        """Forward pass through nodes (aka logic/computation performed 
        at every call).
        """
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. Note the inputs here represent the hidden states 
        # in the model. 
        # 
        # NOTE: Scalar weighting expects atleast 3 dims, so that it can flatten 
        # the 3rd dim (which might be a hidden dim??)
        pooled_output = self.scalar_reweighting(inputs).squeeze(2)
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class LSTMEncoder(nn.Module):
    """"""

    def __init__(self):
        super(LSTMEncoder, self).__init__()
        