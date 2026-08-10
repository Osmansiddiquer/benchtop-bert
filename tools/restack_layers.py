"""Rebuild the layer stack from an explicit order, dropping and inserting layers.

Motivation: layer 4 was measured worth -0.007 nats after two different initialisations
(a stacked copy of L2 for 9,600 steps, then a zero-init graft for 3,800 more). The
pairwise ablation showed nothing was covering for it -- marginal(4|j) stayed at zero for
every j -- so it was not redundant, it simply had no work available at that position.

This tool removes it and inserts a fresh layer at a DIFFERENT position, on the theory
that position, not initialisation, was the binding constraint: a new top layer sits
next to the MLM head, which can use it directly, rather than mid-stack where L1/L2/L5's
existing redundancy already covers the available work.

    --order 0,1,2,3,5,NEW

means: keep layers 0-3, move old layer 5 into slot 4, and put a freshly initialised
layer in slot 5. Integers name source layers; NEW means standard random init, exactly
what the model constructor produces -- no stacking, no zero-init.

Unlike the zero-init graft this is NOT function-preserving, and it is not meant to be.
A random post-LN block on top scrambles the final representation the head reads, so the
loss WILL spike. The tool measures the jump so the size of the debt is known going in.

Optimiser state: any slot whose contents changed identity must have its Adam moments
dropped on resume, or Adam divides fresh gradients by moments describing different
weights. The tool prints the exact --reset-opt-layers list to use.

    python tools/restack_layers.py --ckpt checkpoints/ckpt_v2/last.pt \
        --order 0,1,2,3,5,NEW --backup checkpoints/ckpt_v2/pre_restack.pt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import shutil

import numpy as np
import torch

from mini_enc_transformer.model_v2.mlm import BertForMaskedLM as BertV2, mask_tokens
from mini_enc_transformer.training.pretrain import build_tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ckpt_v2/last.pt")
    p.add_argument("--out", default=None, help="defaults to --ckpt, i.e. in place")
    p.add_argument("--backup", default=None)
    p.add_argument("--order", default="0,1,2,3,5,NEW",
                   help="new stack, one entry per slot: an int names the source layer, "
                        "NEW means standard random init")
    p.add_argument("--seqs", type=int, default=64)
    p.add_argument("--corpora", default="cosmopedia,ultrafineweb_en,tinystories")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def fixed_batch(names, seqs):
    per = max(1, seqs // len(names))
    chunks = []
    for nm in names:
        man = json.load(open(f"data/{nm}.manifest.json"))
        mm = np.memmap(f"data/{nm}.bin", dtype=np.uint16, mode="r",
                       shape=(man["target_tokens"],))
        off = 11_000_000 % max(1, man["tokens_written"] - per * 128 - 1)
        chunks.append(torch.from_numpy(
            np.asarray(mm[off:off + per * 128]).astype(np.int64)).view(-1, 128))
    return torch.cat(chunks)


@torch.no_grad()
def loss_on(model, masked, labels, bs=16):
    tot, n = 0.0, 0
    for i in range(0, masked.shape[0], bs):
        out = model(masked[i:i + bs], labels[i:i + bs])
        tot += float(out["loss"]) * masked[i:i + bs].shape[0]
        n += masked[i:i + bs].shape[0]
    return tot / max(1, n)


def main():
    a = parse_args()
    order = [None if s.strip().upper() == "NEW" else int(s) for s in a.order.split(",")]
    out_path = a.out or a.ckpt
    tk, ids = build_tokenizer("allenai/OLMo-1B-hf")

    ck = torch.load(a.ckpt, map_location="cpu")
    sd = ck["model"]
    n_src = 1 + max(int(k.split(".")[2]) for k in sd
                    if k.startswith("encoder.encoder_blocks."))
    d_ff = sd["encoder.encoder_blocks.0.feed_forward.l1"].shape[1]
    kw = dict(vocab_size=ids["vocab_size"], d_model=768, d_k=64, d_v=64, n_heads=4,
              d_ff=d_ff, pad_id=ids["pad_id"], d_embed=128)

    src = BertV2(n_layers=n_src, **kw)
    src.load_state_dict(sd)
    src.eval()
    print(f"[restack] {a.ckpt}  step {ck.get('step','?')}  {n_src}L d_ff={d_ff}", flush=True)

    torch.manual_seed(1234)
    block = fixed_batch([s.strip() for s in a.corpora.split(",") if s.strip()], a.seqs)
    masked, labels = mask_tokens(block, ids["mask_id"], ids["vocab_size"],
                                 special_token_ids=ids["special_ids"],
                                 pad_token_id=ids["pad_id"], mlm_probability=0.15,
                                 span_min=1, span_max=1)
    before = loss_on(src, masked, labels)
    print(f"[restack] loss before   {before:.4f}", flush=True)

    torch.manual_seed(a.seed)
    dst = BertV2(n_layers=len(order), **kw)            # fresh blocks = standard init
    with torch.no_grad():
        dst.encoder.embedding.load_state_dict(src.encoder.embedding.state_dict())
        dst.encoder.final_norm.load_state_dict(src.encoder.final_norm.state_dict())
        dst.mlm_head.dense.load_state_dict(src.mlm_head.dense.state_dict())
        dst.mlm_head.norm.load_state_dict(src.mlm_head.norm.state_dict())
        dst.mlm_head.decoder.bias.copy_(src.mlm_head.decoder.bias)
        for slot, srcl in enumerate(order):
            if srcl is None:
                print(f"[restack] slot {slot} <- NEW (standard random init)", flush=True)
                continue
            dst.encoder.encoder_blocks[slot].load_state_dict(
                src.encoder.encoder_blocks[srcl].state_dict())
            note = "" if srcl == slot else f"   (moved from {srcl})"
            print(f"[restack] slot {slot} <- source layer {srcl}{note}", flush=True)
    dst.mlm_head.tie_to(dst.encoder.embedding.weight)
    dst.eval()

    after = loss_on(dst, masked, labels)
    print(f"[restack] loss after    {after:.4f}   delta {after - before:+.4f} nats",
          flush=True)

    # Any slot not holding the same layer it held before has changed identity, so its
    # saved Adam moments describe different weights and must be dropped.
    changed = [slot for slot, srcl in enumerate(order) if srcl != slot]
    dropped = sorted(set(range(n_src)) - {s for s in order if s is not None})

    if a.backup and out_path == a.ckpt:
        shutil.copy2(a.ckpt, a.backup)
        print(f"[restack] backed up original -> {a.backup}", flush=True)

    ck["model"] = dst.state_dict()
    ck["restack"] = {"order": a.order, "dropped_layers": dropped,
                     "loss_before": round(before, 4), "loss_after": round(after, 4),
                     "delta": round(after - before, 4), "at_step": ck.get("step")}
    tmp = out_path + ".tmp"
    torch.save(ck, tmp)
    os.replace(tmp, out_path)
    print(f"\n[restack] wrote {out_path}")
    print(f"[restack] dropped layer(s): {dropped}")
    print(f"[restack] NEXT: resume with --reset-opt-layers "
          f"{','.join(str(c) for c in changed)}   (slots whose contents changed identity)")


if __name__ == "__main__":
    main()
