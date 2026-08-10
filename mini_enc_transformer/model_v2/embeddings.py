from mini_enc_transformer.model_v2.utils import initialize_normal_torch_weights
import math 

import torch
from torch import nn


# tokenizer = AutoTokenizer.from_pretrained("tokenizer/tok-16k")

EMBEDDING_DIM = 768

class SlowInputEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, dtype = torch.float32):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = embedding_dim
        self.dtype = dtype
        self.w = self.initialize_weights()
        assert self.w.requires_grad == self.w.is_leaf == True

    def initialize_weights(self):
        return initialize_normal_torch_weights(self.vocab_size, self.d_model)

    def forward(self, input_idx) -> torch.Tensor:
        # input_ids: (batch_size, seq_len (token_ids))
        # output: (batch_size, seq_len, embedding_dim)

        if input_idx.ndim != 2:
            raise ValueError("expected input of shape: N x Seq_len")

        output = self.w[input_idx] * math.sqrt(self.d_model) # from attention is all you need.

        assert output.requires_grad == True
        assert output.shape == (input_idx.shape[0], input_idx.shape[1], self.d_model), f"expected output shape {(input_idx.shape[0], input_idx.shape[1], self.d_model)}, got {output.shape}"

        return output


# optimized version of the embedding layer that uses nn.Embedding for better performance
import math

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """Token ids -> d_model vectors, with an ALBERT-style factorized embedding.

    Takes (B, T) int64 token ids directly. A one-hot matmul computes the same
    thing but materializes a (B, T, vocab_size) tensor -- 0.5 GB per micro-batch
    at B=32, T=128, V=32k -- and burns 0.21 TFLOPs doing a matmul whose only
    effect is to select rows. nn.Embedding is that row selection, for free.

    Factorization (Lan et al., ALBERT): instead of a V x d_model table, use a
    V x d_embed lookup (d_embed << d_model) followed by a d_embed -> d_model
    projection. At V=50281, d_model=768 this cuts the table from V*d_model =
    38.6M params to V*d_embed + d_embed*d_model = 6.5M at d_embed=128 -- the
    lookup stops dominating the model, and the Adam state it carries drops from
    ~460MB to ~78MB (fp32), which matters on a 4GB GPU. Nearly quality-neutral:
    ALBERT's quality regression came from cross-layer sharing, not this.

    The sqrt(d_model) scaling is not decoration. Sinusoidal PE has fixed norm
    sqrt(d_model / 2) = 19.6 at d_model=768, while a projected token vector
    initialized small has a far smaller norm. Added directly, position would
    outweigh token identity; scaling by sqrt(d_model) puts the token vector on
    the same order as the PE it is added to.
    """

    def __init__(self, vocab_size: int, d_model: int, d_embed: int = 128, pad_id = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_embed = d_embed
        # V x d_embed lookup ...
        self.word_emb = nn.Embedding(vocab_size, d_embed, padding_idx=pad_id)
        nn.init.normal_(self.word_emb.weight, mean=0.0, std=0.02)
        if pad_id is not None:
            with torch.no_grad():
                self.word_emb.weight[pad_id].zero_()
        # ... then a (bias-free) d_embed -> d_model projection into the residual stream.
        self.emb_proj = nn.Linear(d_embed, d_model, bias=False)
        nn.init.normal_(self.emb_proj.weight, mean=0.0, std=0.02)

    @property
    def weight(self):
        # The V x d_embed lookup, exposed so the MLM head can tie its decoder to
        # it (the head projects d_model -> d_embed before this shared matrix).
        return self.word_emb.weight

    def forward(self, ids: torch.Tensor):
        if ids.dtype not in (torch.long, torch.int):
            raise TypeError(f"expected integer token ids, got dtype {ids.dtype}")
        if ids.dim() != 2:
            raise ValueError(f"expected (B, T) token ids, got {tuple(ids.shape)}")
        return self.emb_proj(self.word_emb(ids)) * math.sqrt(self.d_model)