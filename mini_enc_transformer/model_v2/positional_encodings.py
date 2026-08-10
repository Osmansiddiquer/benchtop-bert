import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as described in "Attention is All You Need" (Vaswani et al., 2017).
    This encoding allows the model to learn the position of tokens in a sequence without relying on learned embeddings, which can be beneficial for generalization to longer sequences.
    The dropout may be applied after this layer to prevent overfitting, especially in smaller datasets.
    """
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # register_buffer stores this tensor as part of the module state,
        # so it is moved with the model to CUDA/CPU and included in state_dict,
        # but it is not treated as a trainable parameter.
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        # input: N x Seq_len x d_model
        # output: N x Seq_len x d_model
        if x.ndim != 3:
            raise ValueError("expected input of shape: N x Seq_len x d_model")

        n, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(
                f"expected last dimension {self.d_model}, got {d_model}"
            )
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        pe = self.pe[:seq_len].to(dtype=x.dtype, device=x.device)
        pe.requires_grad = False  # Ensure that the positional encoding is not trainable
        return x + pe.unsqueeze(0)
        


class BinaryPositionalEncoding(nn.Module):
    """ 
    WHY THIS IS BAD: Binary positional encoding is not continuous, 
    so the model has a hard time understanding that it is in binary sequence
    and to treat 1,-1, -1, -1 and -1, 1, 1, 1 as neighbours. 
    Cosine similarity of these two vectors will be really low. 
    The model will have to learn hamming distance for this to be useful.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        # input: N x Seq_len X d_model
        # output: N x Seq_len X d_model
        # oscilating between -1 and 1, to keep it centered.

        if x.ndim != 3:
            raise ValueError("expected input of shape: N x Seq_len x d_model")

        n, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(
                f"expected last dimension {self.d_model}, got {d_model}"
            )

        positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        bit_indices = torch.arange(d_model, device=x.device, dtype=torch.long)

        positions = positions.view(1, seq_len, 1)
        bit_indices = bit_indices.view(1, 1, d_model)

        powers = torch.pow(torch.tensor(2, device=x.device), bit_indices)
        binary_bits = ((positions // powers) % 2).to(dtype=x.dtype)
        binary_encoding = binary_bits * 2.0 - 1.0

        assert binary_encoding.shape == (1, seq_len, d_model)  # no need to repeat for batch size, broadcasting will handle it

        return x + binary_encoding
        
