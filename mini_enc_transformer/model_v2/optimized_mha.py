from mini_enc_transformer.model_v2.utils import initialize_normal_torch_weights
import warnings
import torch
import torch.nn as nn

class OptimizedMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int, mask: bool = False, d_out = None):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.use_mask = mask

        if d_model % n_heads != 0:
            warnings.warn(f"d_model ({d_model}) is not divisible by n_heads ({n_heads}).")

        self.d_out = d_out if d_out is not None else d_model

        # Combine all heads into single projection weights for speed
        self.W_q = initialize_normal_torch_weights(d_model, n_heads * d_k)
        self.W_k = initialize_normal_torch_weights(d_model, n_heads * d_k)
        self.W_v = initialize_normal_torch_weights(d_model, n_heads * d_v)
        self.W_O = initialize_normal_torch_weights(n_heads * d_v, self.d_out)

    def forward(self, Z: torch.Tensor, get_attention_list: bool = False):
        # Z shape: [B, T, d_model]
        B, T, _ = Z.shape

        # 1. Linear projections: [B, T, n_heads * d]
        # 2. Reshape & Transpose to batch heads: [B, n_heads, T, d]
        Q = (Z @ self.W_q).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = (Z @ self.W_k).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = (Z @ self.W_v).view(B, T, self.n_heads, self.d_v).transpose(1, 2)

        if get_attention_list:
            # Fallback path if explicit attention maps are requested (Slower, heavy memory)
            E = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
            if self.use_mask:
                attn_mask = torch.triu(torch.ones(T, T, device=Z.device), diagonal=1).bool()
                E = E.masked_fill(attn_mask, float('-inf'))
            A = torch.nn.functional.softmax(E, dim=-1)
            out = A @ V  # [B, n_heads, T, d_v]
            
            # Reshape back to [B, T, n_heads * d_v]
            out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.d_v)
            out = out @ self.W_O
            
            # Split attention maps into a list of heads to preserve your API
            A_list = [A[:, i, :, :] for i in range(self.n_heads)]
            return A_list, out

        # Fast path: Leverages FlashAttention/Memory-Efficient Attention
        # PyTorch expects shapes: [B, n_heads, T, d]
        out = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, 
            is_causal=self.use_mask
        ) # Output: [B, n_heads, T, d_v]
        
        # Reshape back to [B, T, n_heads * d_v] and project output
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.d_v)
        return out @ self.W_O