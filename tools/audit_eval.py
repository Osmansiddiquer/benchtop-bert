"""Audit the eval: is the reported val loss measuring what it claims to?

Checks, in the order they usually fail:

1. TRAIN/VAL OVERLAP. The split is contiguous by block with val taken from the tail, so
   within a run it is clean. The risk is CROSS-RUN: `limit_tokens` changes where the
   boundary falls, so a val range under one limit can sit inside the train range under
   another -- and this model inherited weights from a run with different limits.
   Reported per corpus, as explicit token ranges.

2. DUPLICATE BLOCKS. Exact-hash match of every val block against a large sample of train
   blocks. Catches boilerplate and near-duplicate documents that make prediction look
   like skill.

3. WHAT IS SCORED. Counts positions with label != IGNORE_INDEX and checks whether any
   special token (EOS, PAD, MASK) is among them. Scoring padding is the classic way to
   deflate a reported loss.

4. AVERAGING. Reports mean-of-batch-means (what evaluate() returns) against the correct
   token-weighted total-loss/total-tokens, so the bias is a number rather than a worry.

Also reports the loss restricted to non-EOS scored positions, and the top scored-token
frequencies -- if a handful of tokens dominate, the loss reflects their predictability.

    python tools/audit_eval.py --data-name "cosmopedia:403451249:403451249,..."
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import hashlib
import json
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_enc_transformer.data.dataset import MixtureDataset, PackedMemmapDataset
from mini_enc_transformer.model_v2.mlm import (BertForMaskedLM as BertV2, mask_tokens,
                                               IGNORE_INDEX)
from mini_enc_transformer.training.pretrain import build_tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ckpt_v2/pre_anneal.pt")
    p.add_argument("--data-name", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--prior-runs", default="ultrafineweb_en:166666667,tinystories:166666667",
                   help="corpus:limit pairs from an EARLIER run whose weights were "
                        "inherited; used to test whether this run's val sits inside "
                        "that run's train")
    p.add_argument("--dup-train-blocks", type=int, default=200000)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def specs_of(data_name, default_limit=None):
    out = []
    for part in [s.strip() for s in data_name.split(",") if s.strip()]:
        b = part.split(":")
        out.append((b[0], float(b[1]) if len(b) > 1 and b[1] else 1.0,
                    int(float(b[2])) if len(b) > 2 and b[2] else default_limit))
    return out


def ranges(out_dir, name, limit, seq_len, val_fraction=0.01):
    man = json.load(open(os.path.join(out_dir, name + ".manifest.json")))
    n = man["tokens_written"] if limit is None else min(limit, man["tokens_written"])
    nb = n // seq_len
    nval = max(1, int(nb * val_fraction))
    ntrain = nb - nval
    return dict(n=n, n_blocks=nb, train=(0, ntrain * seq_len),
                val=(ntrain * seq_len, nb * seq_len))


def main():
    a = parse_args()
    tk, ids = build_tokenizer("allenai/OLMo-1B-hf")
    specs = specs_of(a.data_name)

    print("=" * 78)
    print("1. TRAIN/VAL RANGES, and overlap with the inherited run's TRAIN")
    print("=" * 78)
    prior = {n: int(l) for n, l in
             (p.split(":") for p in a.prior_runs.split(",") if p.strip())}
    contaminated = []
    for name, _, limit in specs:
        r = ranges(a.data_dir, name, limit, a.seq_len)
        line = (f"  {name:<18} n={r['n']/1e6:7.1f}M  train=[0, {r['train'][1]/1e6:.1f}M)"
                f"  val=[{r['val'][0]/1e6:.1f}M, {r['val'][1]/1e6:.1f}M)")
        if name in prior:
            pr = ranges(a.data_dir, name, prior[name], a.seq_len)
            overlap = not (r["val"][0] >= pr["train"][1] or r["val"][1] <= pr["train"][0])
            line += f"\n      prior run train=[0, {pr['train'][1]/1e6:.1f}M)  -> " + \
                    ("*** VAL INSIDE PRIOR TRAIN: CONTAMINATED ***" if overlap else "clean")
            if overlap:
                contaminated.append(name)
        print(line)

    dss = [PackedMemmapDataset(a.data_dir, n, a.seq_len, "val", limit_tokens=l)
           for n, _, l in specs]
    val = dss[0] if len(dss) == 1 else MixtureDataset(dss, [w for _, w, _ in specs])
    tr_dss = [PackedMemmapDataset(a.data_dir, n, a.seq_len, "train", limit_tokens=l)
              for n, _, l in specs]

    print()
    print("=" * 78)
    print("2. DUPLICATE BLOCKS (val block byte-identical to a sampled train block)")
    print("=" * 78)
    seen = set()
    rng = np.random.default_rng(0)
    for ds in tr_dss:
        k = min(a.dup_train_blocks // len(tr_dss), len(ds))
        for i in rng.choice(len(ds), size=k, replace=False):
            seen.add(hashlib.blake2b(ds[int(i)].numpy().tobytes(), digest_size=8).digest())
    loader = DataLoader(val, batch_size=a.micro_batch, shuffle=False)
    dup = tot_blocks = 0
    for bi, block in enumerate(loader):
        if bi >= a.batches:
            break
        for row in block:
            tot_blocks += 1
            if hashlib.blake2b(row.numpy().tobytes(), digest_size=8).digest() in seen:
                dup += 1
    print(f"  {dup}/{tot_blocks} val blocks matched one of {len(seen):,} sampled train "
          f"blocks  ({100*dup/max(1,tot_blocks):.2f}%)")

    print()
    print("=" * 78)
    print("3/4. WHAT IS SCORED, and how it is averaged")
    print("=" * 78)
    sd = torch.load(a.ckpt, map_location="cpu")["model"]
    nl = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("encoder.encoder_blocks."))
    d_ff = sd["encoder.encoder_blocks.0.feed_forward.l1"].shape[1]
    model = BertV2(vocab_size=ids["vocab_size"], d_model=768, d_k=64, d_v=64, n_heads=4,
                   n_layers=nl, d_ff=d_ff, pad_id=ids["pad_id"], d_embed=128)
    model.load_state_dict(sd)
    model.eval().to(a.device)

    eos, pad, msk = ids.get("eos_id", tk.eos_token_id), ids["pad_id"], ids["mask_id"]
    specials = set(ids.get("special_ids") or [])
    g = torch.Generator().manual_seed(1234)
    batch_means, sum_loss, n_tok, n_pos = [], 0.0, 0, 0
    sum_loss_noeos, n_tok_noeos = 0.0, 0
    scored_tokens = Counter()
    special_scored = 0
    loader = DataLoader(val, batch_size=a.micro_batch, shuffle=False)
    for bi, block in enumerate(loader):
        if bi >= a.batches:
            break
        masked, labels = mask_tokens(block, msk, ids["vocab_size"],
                                     special_token_ids=ids["special_ids"],
                                     pad_token_id=pad, generator=g, span_min=1, span_max=1)
        masked, labels = masked.to(a.device), labels.to(a.device)
        out = model(masked, labels)
        batch_means.append(float(out["loss"]))
        sel = labels != IGNORE_INDEX
        lab = labels[sel]
        lg = out["logits"][sel]
        per_tok = F.cross_entropy(lg.float(), lab, reduction="none")
        sum_loss += float(per_tok.sum()); n_tok += int(sel.sum())
        n_pos += labels.numel()
        keep = lab != eos
        sum_loss_noeos += float(per_tok[keep].sum()); n_tok_noeos += int(keep.sum())
        scored_tokens.update(lab.tolist())
        special_scored += int(sum((int(t) in specials) for t in lab.tolist()))

    mean_of_means = sum(batch_means) / len(batch_means)
    token_weighted = sum_loss / max(1, n_tok)
    print(f"  scored positions        {n_tok:,} of {n_pos:,}  ({100*n_tok/n_pos:.2f}% "
          f"of all positions)")
    print(f"  special tokens scored   {special_scored}  (EOS={eos} PAD={pad} MASK={msk})")
    print(f"  EOS among scored        {scored_tokens.get(eos,0):,} "
          f"({100*scored_tokens.get(eos,0)/max(1,n_tok):.2f}%)")
    print()
    print(f"  mean-of-batch-means     {mean_of_means:.4f}   <- what evaluate() reports")
    print(f"  token-weighted          {token_weighted:.4f}   <- correct")
    print(f"  bias                    {mean_of_means - token_weighted:+.4f} nats")
    print(f"  token-weighted, no EOS  {sum_loss_noeos/max(1,n_tok_noeos):.4f}")
    print()
    top = scored_tokens.most_common(8)
    print("  most-scored tokens (share of scored positions):")
    for t, c in top:
        s = tk.decode([t]).replace("\n", "\\n")
        print(f"    {t:>6} {repr(s):<14} {100*c/max(1,n_tok):5.2f}%")
    print(f"  top-8 share {100*sum(c for _, c in top)/max(1,n_tok):.1f}%   "
          f"distinct scored types {len(scored_tokens):,}")

    if contaminated:
        print(f"\n  ==> CONTAMINATED CORPORA: {contaminated}")


if __name__ == "__main__":
    main()
