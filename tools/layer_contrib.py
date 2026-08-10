"""What is each layer actually worth? Ablate it and measure the loss it was buying.

The question this answers: is a layer doing work, or is the rest of the stack routing
around it? Every other diagnostic here (never_pos, utilisation, head entropy, head_sim)
is a proxy for that. This is the thing itself.

Ablation zeroes a layer's two OUTPUT projections -- mha.W_O and feed_forward.l2/b2 --
which is exactly the counterfactual tools/graft_l4.py creates. The block still applies
its layer norms (this model is post-LN, so a zeroed block computes ln2(ln1(x)), not the
identity), which makes the number directly comparable to the graft's measured delta.

Read it as: "removing this layer costs N nats." A layer worth ~0.00 is decorative --
the stack has converged to a solution that does not use it, and no amount of learning
rate on that layer alone will change that, because the gradient it receives is the
gradient of a loss surface that is already at an optimum without it.

    python tools/layer_contrib.py --ckpt checkpoints/ckpt_v2/last.pt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy
import json

import numpy as np
import torch

from mini_enc_transformer.model.mlm import BertForMaskedLM as BertV1
from mini_enc_transformer.model_v2.mlm import BertForMaskedLM as BertV2, mask_tokens
from mini_enc_transformer.training.pretrain import build_tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ckpt_v2/last.pt")
    p.add_argument("--seqs", type=int, default=64)
    p.add_argument("--corpora", default="cosmopedia,ultrafineweb_en,tinystories")
    p.add_argument("--limits",
                   default="cosmopedia:403451249,ultrafineweb_en:256832885,tinystories:73377468",
                   help="corpus:limit_tokens matching the TRAINING run, so the val "
                        "boundary is computed identically. Wrong limits => wrong split.")
    p.add_argument("--span-min", type=int, default=1, help="1/1 = scattered eval masking")
    p.add_argument("--span-max", type=int, default=1)
    p.add_argument("--history", default="logs/layer_contrib.jsonl")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--target", type=int, default=None,
                   help="also run the PAIRWISE redundancy test for this layer: for every "
                        "other layer j, marginal(t|j) = cost({t,j}) - cost({j}). Solo "
                        "ablation understates a layer that another layer covers for; if "
                        "marginal(t|j) greatly exceeds t's solo cost, t is redundant with j.")
    p.add_argument("--full", action="store_true",
                   help="full pairwise matrix over all layer pairs, not just --target")
    return p.parse_args()


def val_offset(man, limit, seq_len=128, val_fraction=0.01):
    """First token of the HELD-OUT range, matching PackedMemmapDataset exactly.

    The dataset splits contiguously by block with val taken from the tail, and the
    boundary depends on limit_tokens -- so a tool that hardcodes an offset silently
    reads TRAINING data. This one previously read at token 11,000,000, which is inside
    the train range of every corpus in this project.
    """
    n = man["tokens_written"] if limit is None else min(limit, man["tokens_written"])
    n_blocks = n // seq_len
    n_train = n_blocks - max(1, int(n_blocks * val_fraction))
    return n_train * seq_len, n_blocks * seq_len


def fixed_batch(names, seqs, limits=None):
    per = max(1, seqs // len(names))
    chunks = []
    for nm in names:
        man = json.load(open(f"data/{nm}.manifest.json"))
        mm = np.memmap(f"data/{nm}.bin", dtype=np.uint16, mode="r",
                       shape=(man["target_tokens"],))
        lo, hi = val_offset(man, (limits or {}).get(nm))
        need = per * 128
        if hi - lo < need:                      # tiny val range: take the last `need`
            lo = max(0, hi - need)
        chunks.append(torch.from_numpy(
            np.asarray(mm[lo:lo + need]).astype(np.int64)).view(-1, 128))
    return torch.cat(chunks)


@torch.no_grad()
def loss_on(model, masked, labels, bs=16):
    tot, n = 0.0, 0
    for i in range(0, masked.shape[0], bs):
        out = model(masked[i:i + bs], labels[i:i + bs])
        tot += float(out["loss"]) * masked[i:i + bs].shape[0]
        n += masked[i:i + bs].shape[0]
    return tot / max(1, n)


def load_any(path, ids):
    sd = torch.load(path, map_location="cpu")["model"]
    nl = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("encoder.encoder_blocks."))
    d_ff = sd["encoder.encoder_blocks.0.feed_forward.l1"].shape[1]
    cls = BertV1 if (nl == 4 and d_ff == 3072) else BertV2
    kw = dict(vocab_size=ids["vocab_size"], d_model=768, d_k=64, d_v=64, n_heads=4,
              n_layers=nl, pad_id=ids["pad_id"], d_embed=128)
    m = cls(**kw) if cls is BertV1 else cls(d_ff=d_ff, **kw)
    m.load_state_dict(sd)
    m.eval()
    return m, nl


def main():
    a = parse_args()
    tk, ids = build_tokenizer("allenai/OLMo-1B-hf")
    model, n_layers = load_any(a.ckpt, ids)
    step = torch.load(a.ckpt, map_location="cpu").get("step", 0)

    torch.manual_seed(1234)
    limits = {k: int(v) for k, v in (x.split(":") for x in a.limits.split(",") if x.strip())}
    block = fixed_batch([s.strip() for s in a.corpora.split(",") if s.strip()], a.seqs, limits)
    masked, labels = mask_tokens(block, ids["mask_id"], ids["vocab_size"],
                                 special_token_ids=ids["special_ids"],
                                 pad_token_id=ids["pad_id"], mlm_probability=0.15,
                                 span_min=a.span_min, span_max=a.span_max)

    base = loss_on(model, masked, labels)

    def cost(layers):
        """Loss increase when every layer in `layers` has its output projections zeroed."""
        abl = copy.deepcopy(model)
        with torch.no_grad():
            for i in layers:
                blk = abl.encoder.encoder_blocks[i]
                blk.mha.W_O.zero_()
                blk.feed_forward.l2.zero_()
                blk.feed_forward.b2.zero_()
        c = loss_on(abl, masked, labels) - base
        del abl
        return c

    rows = [dict(layer=i, cost=cost([i])) for i in range(n_layers)]
    solo = {r["layer"]: r["cost"] for r in rows}

    # ---- pairwise redundancy -------------------------------------------------------
    # Solo ablation systematically understates a layer that another layer covers for:
    # zero L4 alone and L5 absorbs the slack, so L4 reads as worthless. Zero both and
    # the difference against cost({j}) alone reveals what L4 was actually holding.
    pairs = []
    if a.full:
        want = [(i, j) for i in range(n_layers) for j in range(i + 1, n_layers)]
    elif a.target is not None:
        want = [(min(a.target, j), max(a.target, j))
                for j in range(n_layers) if j != a.target]
    else:
        want = []
    for i, j in want:
        cij = cost([i, j])
        pairs.append(dict(i=i, j=j, cost=cij,
                          marg_i=cij - solo[j],       # what i is worth once j is gone
                          marg_j=cij - solo[i]))      # and vice versa

    with open(a.history, "a") as f:
        f.write(json.dumps(dict(ckpt=a.ckpt, step=step, base=round(base, 4),
                                layers=rows, pairs=pairs)) + "\n")

    if a.quiet:
        print(f"contrib step {step} base_loss {base:.4f} (nats lost if the layer is removed)")
        print("  " + "  ".join(f"L{r['layer']}={r['cost']:+.3f}" for r in rows))
        return

    print(f"[layer-contrib] {a.ckpt}  step {step}   base loss {base:.4f}\n")
    print(f"{'layer':6} {'ablated loss':>13} {'cost (nats)':>12}  {'':4} share")
    tot = sum(max(0.0, r["cost"]) for r in rows) or 1.0
    for r in rows:
        bar = "#" * int(40 * max(0.0, r["cost"]) / max(x["cost"] for x in rows))
        print(f"{r['layer']:<6} {base + r['cost']:13.4f} {r['cost']:+12.4f}  "
              f"{100*max(0.0, r['cost'])/tot:4.0f}% {bar}")
    print("\n  cost ~0.00 => nothing DOWNSTREAM needs this layer -- but see the pairwise "
          "test below\n  before concluding it is useless: a layer another layer covers "
          "for reads the same way.")

    if pairs and a.target is not None and not a.full:
        t = a.target
        print(f"\n  redundancy test for L{t}   (solo cost {solo[t]:+.4f})\n")
        print(f"  {'drop with':>10} {'cost({j})':>11} {'cost({t,j})':>13} "
              f"{'marginal(t|j)':>15} {'vs solo':>9}")
        for p in pairs:
            j = p["j"] if p["i"] == t else p["i"]
            marg = p["marg_i"] if p["i"] == t else p["marg_j"]
            print(f"  {'L%d' % j:>10} {solo[j]:11.4f} {p['cost']:13.4f} "
                  f"{marg:15.4f} {marg - solo[t]:+9.4f}")
        best = max(pairs, key=lambda p: (p["marg_i"] if p["i"] == t else p["marg_j"]))
        bj = best["j"] if best["i"] == t else best["i"]
        bm = best["marg_i"] if best["i"] == t else best["marg_j"]
        print(f"\n  marginal(L{t}|Lj) >> solo => L{t} is redundant WITH Lj and the solo "
              f"number was misleading.\n  largest: L{bj}, marginal {bm:+.4f} vs solo "
              f"{solo[t]:+.4f} (lift {bm - solo[t]:+.4f})")
    elif pairs:
        print(f"\n  full pairwise matrix: marginal(i|j) = cost({{i,j}}) - cost({{j}})\n")
        print("  " + "".join(f"{'L%d' % j:>9}" for j in range(n_layers)) + "   <- given j")
        for i in range(n_layers):
            cells = []
            for j in range(n_layers):
                if i == j:
                    cells.append(f"{'--':>9}")
                else:
                    p = next(p for p in pairs if {p["i"], p["j"]} == {i, j})
                    cells.append(f"{(p['marg_i'] if p['i'] == i else p['marg_j']):9.3f}")
            print(f"L{i}" + "".join(cells) + f"   solo {solo[i]:+.3f}")
        print("\n  row i = what layer i is still worth once layer j is also gone")


if __name__ == "__main__":
    main()
