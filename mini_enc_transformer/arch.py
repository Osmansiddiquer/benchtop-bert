"""Detect a checkpoint's architecture and build the matching encoder.

Downstream scripts (SST-2 fine-tune, benchmark) used to hardcode the v1 shape --
4 layers, d_ff 3072 -- because that was the only architecture when they were written.
Handing them a v2 checkpoint (6 layers, d_ff 1792, GELU) fails with a wall of
`size mismatch for encoder_blocks.N.feed_forward.l1` and nothing else.

The two Encoder classes take identical constructor arguments; they differ only in the
block internals (v1: ReLU and a hardcoded d_model*4 FFN, so it silently ignores d_ff;
v2: GELU and honours d_ff). So picking the class and passing the detected d_ff is the
whole fix, and it belongs in one place rather than copied into each caller.
"""
import torch

from mini_enc_transformer.model.encoder import Encoder as EncoderV1
from mini_enc_transformer.model_v2.encoder import Encoder as EncoderV2


def encoder_subtree(sd):
    """The encoder weights, whether the dict is a full BertForMaskedLM, a fine-tune
    checkpoint (encoder + head), or a bare encoder."""
    sub = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    if sub:
        return sub
    return {k: v for k, v in sd.items()
            if not k.startswith(("mlm_head.", "head.", "predictor."))}


def detect_arch(sd):
    """n_layers and d_ff read off the tensors, plus which model package they belong to.

    v1 is exactly (4 layers, d_ff 3072); everything else is v2. That rule is what the
    rest of the tooling already uses, so keep it identical here -- two different
    detection rules in one repo is worse than one imperfect rule.
    """
    enc = encoder_subtree(sd)
    keys = [k for k in enc if k.startswith("encoder_blocks.")]
    if not keys:
        raise ValueError("state dict has no encoder_blocks.* entries")
    n_layers = 1 + max(int(k.split(".")[1]) for k in keys)
    d_ff = enc["encoder_blocks.0.feed_forward.l1"].shape[1]
    is_v2 = not (n_layers == 4 and d_ff == 3072)
    return dict(n_layers=n_layers, d_ff=d_ff, is_v2=is_v2,
                act="gelu" if is_v2 else "relu")


def build_encoder(sd, ids, d_model=768, d_k=64, d_v=64, n_heads=4, d_embed=128):
    """Returns (encoder, arch) with the encoder shaped to match `sd` but NOT yet loaded."""
    a = detect_arch(sd)
    cls = EncoderV2 if a["is_v2"] else EncoderV1
    enc = cls(ids["vocab_size"], d_model, d_k, d_v, n_heads, a["n_layers"],
              d_ff=a["d_ff"], pad_id=ids["pad_id"], d_embed=d_embed)
    return enc, a


def load_checkpoint_encoder(path, ids, **kw):
    """Build and load in one step. Returns (encoder, arch, full_state_dict)."""
    sd = torch.load(path, map_location="cpu")["model"]
    enc, a = build_encoder(sd, ids, **kw)
    missing, _ = enc.load_state_dict(encoder_subtree(sd), strict=False)
    if missing:
        raise RuntimeError(f"checkpoint is missing encoder weights: {missing[:5]}")
    return enc, a, sd
