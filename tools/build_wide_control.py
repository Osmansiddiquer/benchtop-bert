"""Control arm for the depth-vs-width question: 4 layers, d_ff 3072, GELU.

v2 spends its budget on depth (6 layers, d_ff 1792). This keeps ckpt3's shape and
changes only the activation, so a matched run isolates the allocation:

    v2       6L x 1792  GELU   27.95M
    control  4L x 3072  GELU   28.73M

Transfer is deliberately minimal -- every weight copies across unchanged and only the
activation function differs. No pruning, no stacking. That makes it the gentlest
possible transfer, which is worth noting when reading the two starting losses.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from mini_enc_transformer.model_v2.mlm import BertForMaskedLM as BertV2
from mini_enc_transformer.training.pretrain import build_tokenizer

tk, ids = build_tokenizer("allenai/OLMo-1B-hf")
src = torch.load("checkpoints/ckpt3/last.pt", map_location="cpu")["model"]
dst = BertV2(vocab_size=ids["vocab_size"], d_model=768, d_k=64, d_v=64, n_heads=4,
             n_layers=4, d_ff=3072, pad_id=ids["pad_id"], d_embed=128)
missing, unexpected = dst.load_state_dict(src, strict=False)
print("  missing :", [m for m in missing][:6] or "none")
print("  unexpected:", [u for u in unexpected][:6] or "none")
dst.mlm_head.tie_to(dst.encoder.embedding.weight)
n = sum(p.numel() for p in dst.parameters())
torch.save({"model": dst.state_dict(), "step": 0, "built_from": "checkpoints/ckpt3/last.pt",
            "arch": {"n_layers": 4, "d_ff": 3072, "act": "gelu"}},
           "checkpoints/ckpt_wide_init.pt")
print(f"  {n/1e6:.3f}M params -> checkpoints/ckpt_wide_init.pt")
