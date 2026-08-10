from mini_enc_transformer.model_v2.utils import initialize_normal_torch_weights
import torch
from torch import nn


class FeedForwardBlock(nn.Module):
    """Position-wise feed-forward block, v2.

    Two changes from v1:

    1. **GELU instead of ReLU.** Measured on ckpt3, layer 0 had 2,755 of 3,072 hidden
       units that never fired on 8,192 tokens -- a dying-ReLU signature, leaving ~317
       live units doing all the work. GELU is smooth and non-zero for negative inputs,
       so units cannot die the same way. BERT and GPT both use GELU for this reason.

    2. **d_ff is actually honoured.** v1 accepted a d_ff argument and then hardcoded
       d_model * 4, so passing --d-ff did nothing at all.
    """

    def __init__(self, d_model, d_ff=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else d_model * 4
        self.l1 = initialize_normal_torch_weights(d_model, self.d_ff)
        self.b1 = nn.Parameter(torch.zeros(self.d_ff))
        self.act = nn.GELU()
        self.l2 = initialize_normal_torch_weights(self.d_ff, d_model)
        self.b2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        assert x.shape[-1] == self.d_model
        return self.act(x @ self.l1 + self.b1) @ self.l2 + self.b2
