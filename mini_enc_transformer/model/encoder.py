from mini_enc_transformer.model.positional_encodings import SinusoidalPositionalEncoding
from mini_enc_transformer.model.blocks import EncoderBlock
from mini_enc_transformer.model.embeddings import InputEmbedding

import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, d_k: int, d_v: int, n_heads: int,
                 n_layers: int, d_ff=None, mask=False, pad_id=None, d_embed: int = 128):
        super().__init__()
        self.d_model = d_model
        self.embedding = InputEmbedding(vocab_size, d_model, d_embed=d_embed, pad_id=pad_id)
        self.pe = SinusoidalPositionalEncoding(d_model)
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(d_model, d_k, d_v, n_heads, d_ff, mask) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, ids: torch.Tensor, output_hidden_states: bool = False):
        # ids: (B, T) int64 token ids -> (B, T, d_model)
        if ids.dim() != 2:
            raise ValueError(f"expected (B, T) token ids, got {tuple(ids.shape)}")

        x = self.pe(self.embedding(ids))
        hidden = [] if output_hidden_states else None
        for block in self.encoder_blocks:
            x = block(x)
            if output_hidden_states:
                hidden.append(x)
        out = self.final_norm(x)
        # Latent-target objectives (data2vec/JEPA) average the top-K block outputs
        # rather than using only the last -- a single layer is a noisier target.
        return (out, hidden) if output_hidden_states else out