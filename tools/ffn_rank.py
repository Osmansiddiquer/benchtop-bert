"""Rank analysis of the FFN blocks -- can d_ff be cut to fund extra depth?

The FFN is 65.7% of this model's parameters (4.72M per layer vs 0.79M for attention),
so it is the only place to find budget for more layers. Two questions, and they are not
the same:

  WEIGHTS   : how much of l1 / l2's 768 available directions actually carry signal.
              Low weight rank means the map is degenerate and could be FACTORISED
              (W ~= UV), which only saves parameters below rank ~614 -- above that the
              two factors cost more than the original matrix.

  ACTIVATIONS: how many of the 3072 hidden units are doing independent work on real
              text. This is the question that decides whether d_ff is oversized, and
              it is the one weights alone cannot answer -- a full-rank weight matrix
              can still produce activations that live in a much smaller subspace.

Effective rank is exp(entropy of the normalised singular-value spectrum): the number
of directions actually in use, not the number mathematically available.

    python tools/ffn_rank.py --ckpt checkpoints/ckpt3/last.pt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import numpy as np
import torch

from mini_enc_transformer.model.mlm import BertForMaskedLM as BertV1
from mini_enc_transformer.model_v2.mlm import BertForMaskedLM as BertV2
from mini_enc_transformer.training.pretrain import build_tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ckpt3/last.pt")
    p.add_argument("--corpus", default="cosmopedia")
    p.add_argument("--seqs", type=int, default=64, help="sequences of 128 tokens to probe with")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


class C:
    d_model, d_k, d_v, n_heads, n_layers, d_embed = 768, 64, 64, 4, 4, 128


def load_any(path, ids):
    """Build whichever architecture the checkpoint holds, read off the tensor shapes."""
    sd = torch.load(path, map_location="cpu")["model"]
    n_layers = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("encoder.encoder_blocks."))
    d_ff = sd["encoder.encoder_blocks.0.feed_forward.l1"].shape[1]
    cls = BertV1 if (n_layers == 4 and d_ff == 3072) else BertV2
    kw = dict(vocab_size=ids["vocab_size"], d_model=768, d_k=64, d_v=64, n_heads=4,
              n_layers=n_layers, pad_id=ids["pad_id"], d_embed=128)
    m = cls(**kw) if cls is BertV1 else cls(d_ff=d_ff, **kw)
    m.load_state_dict(sd)
    m.eval()
    act = "relu" if cls is BertV1 else "gelu"
    return m, n_layers, d_ff, act, sd


def eff_rank(sv):
    p = sv / sv.sum().clamp_min(1e-12)
    return float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))


def energy_at(sv, frac):
    """Fraction of squared spectral energy captured by the top `frac` of directions."""
    e = (sv ** 2).cumsum(0) / (sv ** 2).sum()
    return float(e[int(frac * len(sv)) - 1])


def rank_for_energy(sv, target):
    e = (sv ** 2).cumsum(0) / (sv ** 2).sum()
    return int((e < target).sum().item()) + 1


def main():
    a = parse_args()
    tk, ids = build_tokenizer("allenai/OLMo-1B-hf")
    model = build_model(C(), ids)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])
    model.eval()

    print(f"[ffn-rank] {a.ckpt}\n")
    print("WEIGHT SPECTRA  (max rank 768 for both matrices)")
    print(f"{'layer':7} {'matrix':7} {'eff_rank':>9} {'r@90%':>7} {'r@99%':>7} {'energy top-25%':>15}")
    weights = {}
    for i, blk in enumerate(model.encoder.encoder_blocks):
        for name in ("l1", "l2"):
            W = getattr(blk.feed_forward, name).detach().float()
            sv = torch.linalg.svdvals(W)
            weights[f"{i}.{name}"] = dict(eff_rank=eff_rank(sv), r90=rank_for_energy(sv, 0.90),
                                          r99=rank_for_energy(sv, 0.99), e25=energy_at(sv, 0.25))
            w = weights[f"{i}.{name}"]
            print(f"{i:<7} {name:7} {w['eff_rank']:9.1f} {w['r90']:7d} {w['r99']:7d} "
                  f"{100*w['e25']:14.1f}%")

    # ---- activations: the question that actually decides d_ff -------------------
    man = json.load(open(f"data/{a.corpus}.manifest.json"))
    mm = np.memmap(f"data/{a.corpus}.bin", dtype=np.uint16, mode="r", shape=(man["target_tokens"],))
    n = a.seqs * 128
    ids_t = torch.from_numpy(np.asarray(mm[10_000_000:10_000_000 + n]).astype(np.int64)).view(-1, 128)

    hidden = {}
    with torch.no_grad():
        x = model.encoder.pe(model.encoder.embedding(ids_t))
        for i, blk in enumerate(model.encoder.encoder_blocks):
            attn = blk.mha(x, False)
            h = blk.layer_norm_1(attn + x)
            ff = blk.feed_forward
            pre = h @ ff.l1 + ff.b1              # (B, T, 3072) before ReLU
            act = ff.relu_1(pre)
            hidden[i] = act.reshape(-1, act.size(-1)).float()
            x = blk.layer_norm_2(ff(h) + h)

    print(f"\nHIDDEN ACTIVATIONS on {a.seqs} sequences ({n:,} tokens), d_ff = 3072")
    print(f"{'layer':7} {'eff_rank':>9} {'r@90%':>7} {'r@99%':>7} {'dead units':>11} {'usable d_ff':>12}")
    for i, H in hidden.items():
        Hc = H - H.mean(0, keepdim=True)
        sv = torch.linalg.svdvals(Hc[:4096])
        dead = int((H.abs().max(0).values < 1e-6).sum())   # ReLU units never firing
        print(f"{i:<7} {eff_rank(sv):9.1f} {rank_for_energy(sv,0.90):7d} "
              f"{rank_for_energy(sv,0.99):7d} {dead:11d} {rank_for_energy(sv,0.99):12d}")

    print("\nreading: r@99% is the width that would preserve 99% of the activation")
    print("variance. If that is well under 3072, d_ff is oversized and the surplus can")
    print("fund extra layers. Factorising the WEIGHTS only pays below rank ~614.")


if __name__ == "__main__":
    main()
