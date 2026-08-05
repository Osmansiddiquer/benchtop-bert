from mini_enc_transformer.model.optimized_mha import OptimizedMultiHeadAttention
from mini_enc_transformer.model.positional_encodings import SinusoidalPositionalEncoding
from mini_enc_transformer.model.normalization import LayerNormalization
from mini_enc_transformer.model.ffn import FeedForwardBlock
import torch
from torch import nn



class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, d_k: int, d_v: int, n_heads: int, d_ff = None, mask = False):
        super().__init__()
        self.mha = OptimizedMultiHeadAttention(d_model, n_heads, d_k, d_v, mask)
        self.layer_norm_1 = LayerNormalization(d_model)
        self.feed_forward = FeedForwardBlock(d_model, d_ff)
        self.layer_norm_2 = LayerNormalization(d_model)

    def forward(self, x: torch.Tensor, get_attention_list=False):
        # x is B x T x d_model. T is max seq length
        # Output is B x T x d_model which is enriched.
        if len(x.shape) != 3 or x.shape[2] != self.mha.d_model:
            raise ValueError(f"x must be of shape B x T x {self.mha.d_model}, but got {x.shape}")

        # Sublayer 1: multi-head self-attention, with a residual around it.
        A_list = None
        if get_attention_list:
            A_list, attn = self.mha(x, True)
        else:
            attn = self.mha(x, False)
        x = self.layer_norm_1(attn + x)

        # Sublayer 2: position-wise FFN, with a residual around *its own* input.
        ff = self.feed_forward(x)
        out = self.layer_norm_2(ff + x)

        assert out.shape == (x.shape[0], x.shape[1], self.mha.d_model), (
            f"Output shape must have been B x T x {self.mha.d_model}, but got {out.shape}"
        )
        if get_attention_list:
            return A_list, out
        else:
            return out



        
        