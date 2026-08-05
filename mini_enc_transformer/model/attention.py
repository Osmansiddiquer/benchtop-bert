import warnings
import torch
import torch.nn as nn
from mini_enc_transformer.model.utils import initialize_normal_torch_weights


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int, d_v: int, d_model: int, mask: bool = False):
        super().__init__()
        self.W_k = nn.Parameter(initialize_normal_torch_weights(d_model, d_k))
        self.W_q = nn.Parameter(initialize_normal_torch_weights(d_model, d_k))
        self.W_v = nn.Parameter(initialize_normal_torch_weights(d_model, d_v))
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.use_mask = mask

    def forward(self, Z: torch.Tensor, get_attention: bool = False):
        if Z.ndim != 3 or Z.shape[2] != self.d_model:
            raise ValueError(f"Z must be of shape B x T x {self.d_model}, but got {Z.shape}")
        
        B, T, _ = Z.shape
        Q = Z @ self.W_q
        K = Z @ self.W_k
        V = Z @ self.W_v

        attn_mask = torch.triu(torch.ones(T, T, device=Z.device), diagonal=1).bool() if self.use_mask else None

        if get_attention:
            E = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
            if attn_mask is not None:
                E = E.masked_fill(attn_mask, float('-inf'))
            A = nn.functional.softmax(E, dim=-1)
            out = A @ V
            return A, out
            
        return nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=self.use_mask)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k: int, d_v: int, d_model: int, n_heads: int, mask: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.mask = mask

        if d_model % n_heads != 0:
            warnings.warn(f"d_model ({d_model}) is not divisible by n_heads ({n_heads}).")

        self.heads = nn.ModuleList([
            ScaledDotProductAttention(d_k, d_v, d_model, mask) for _ in range(n_heads)
        ])
        self.W_O = nn.Parameter(initialize_normal_torch_weights(n_heads * d_v, d_model))

    def forward(self, Z: torch.Tensor, get_attention_list: bool = False):
        if Z.ndim != 3 or Z.shape[2] != self.d_model:
            raise ValueError(f"Z must be of shape B x T x {self.d_model}, but got {Z.shape}")

        A_list, out_list = [], []
        for head in self.heads:
            res = head(Z, get_attention=get_attention_list)
            if get_attention_list:
                A, out = res
                A_list.append(A)
                out_list.append(out)
            else:
                out_list.append(res)

        out = torch.cat(out_list, dim=2) @ self.W_O
        
        if get_attention_list:
            return A_list, out
        return out