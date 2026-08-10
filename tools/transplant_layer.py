"""Copy specific layers from one checkpoint into another, in place.

Built for one job: layer 4 was trained alone against a frozen stack, where it reached an
ablation cost of +0.060 nats against the -0.003 it converged to under joint training.
Every other parameter was frozen during that run, so the two checkpoints differ in
exactly the twelve tensors of layer 4 -- verified before writing, not assumed.

That makes the transplant surgical. The recipient keeps its optimiser moments, schedule
position, RNG state and step count; only the donated layer changes, and only its moments
need dropping (--reset-opt-layers), because every other layer's weights never moved and
their moments remain exactly valid.

The point of the exercise: L4 previously failed because it arrived useless into a stack
that had already partitioned the work between L1/L2/L5. Now it arrives useful. Whether
its contribution survives joint training answers whether routing-around is the stable
attractor or merely the basin the old initialisation fell into.

    python tools/transplant_layer.py \
        --src checkpoints/ckpt_v2_l4only/last.pt \
        --dst checkpoints/ckpt_v2/last.pt --layers 4 \
        --backup checkpoints/ckpt_v2/pre_transplant.pt
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
    p.add_argument("--src", required=True, help="donor checkpoint")
    p.add_argument("--dst", required=True, help="recipient, modified in place")
    p.add_argument("--layers", default="4")
    p.add_argument("--backup", default=None)
    p.add_argument("--seqs", type=int, default=64)
    p.add_argument("--corpora", default="cosmopedia,ultrafineweb_en,tinystories")
    p.add_argument("--allow-other-diffs", action="store_true",
                   help="proceed even if the checkpoints differ outside --layers")
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
def loss_of(sd, ids, masked, labels):
    n_layers = 1 + max(int(k.split(".")[2]) for k in sd
                       if k.startswith("encoder.encoder_blocks."))
    d_ff = sd["encoder.encoder_blocks.0.feed_forward.l1"].shape[1]
    m = BertV2(vocab_size=ids["vocab_size"], d_model=768, d_k=64, d_v=64, n_heads=4,
               n_layers=n_layers, d_ff=d_ff, pad_id=ids["pad_id"], d_embed=128)
    m.load_state_dict(sd)
    m.eval()
    tot, n = 0.0, 0
    for i in range(0, masked.shape[0], 16):
        out = m(masked[i:i + 16], labels[i:i + 16])
        tot += float(out["loss"]) * masked[i:i + 16].shape[0]
        n += masked[i:i + 16].shape[0]
    return tot / max(1, n)


def main():
    a = parse_args()
    layers = [int(x) for x in a.layers.split(",") if x.strip()]
    tk, ids = build_tokenizer("allenai/OLMo-1B-hf")

    src = torch.load(a.src, map_location="cpu")
    dst = torch.load(a.dst, map_location="cpu")
    S, D = src["model"], dst["model"]
    print(f"[transplant] donor     {a.src}  step {src.get('step','?')}")
    print(f"[transplant] recipient {a.dst}  step {dst.get('step','?')}", flush=True)

    pref = tuple(f"encoder.encoder_blocks.{i}." for i in layers)
    diff = [k for k in D if not torch.equal(D[k], S[k])]
    outside = [k for k in diff if not k.startswith(pref)]
    print(f"[transplant] tensors differing: {len(diff)}, "
          f"outside layers {layers}: {len(outside)}", flush=True)
    if outside and not a.allow_other_diffs:
        print(f"[transplant] ABORT: the checkpoints differ outside the donated layers, so "
              f"this is not a clean swap:\n    {outside[:6]}\n  Re-run with "
              f"--allow-other-diffs to accept.", flush=True)
        sys.exit(1)

    torch.manual_seed(1234)
    block = fixed_batch([s.strip() for s in a.corpora.split(",") if s.strip()], a.seqs)
    masked, labels = mask_tokens(block, ids["mask_id"], ids["vocab_size"],
                                 special_token_ids=ids["special_ids"],
                                 pad_token_id=ids["pad_id"], mlm_probability=0.15,
                                 span_min=1, span_max=1)

    before = loss_of(D, ids, masked, labels)
    moved = [k for k in D if k.startswith(pref)]
    for k in moved:
        D[k] = S[k].clone()
    after = loss_of(D, ids, masked, labels)
    print(f"[transplant] moved {len(moved)} tensors for layer(s) {layers}")
    print(f"[transplant] recipient loss  {before:.4f} -> {after:.4f}   "
          f"delta {after - before:+.4f} nats", flush=True)

    if a.backup:
        shutil.copy2(a.dst, a.backup)
        print(f"[transplant] backed up recipient -> {a.backup}", flush=True)

    dst["model"] = D
    dst["transplant"] = {"src": a.src, "src_step": src.get("step"), "layers": layers,
                         "loss_before": round(before, 4), "loss_after": round(after, 4),
                         "delta": round(after - before, 4), "at_step": dst.get("step")}
    tmp = a.dst + ".tmp"
    torch.save(dst, tmp)
    os.replace(tmp, a.dst)
    print(f"[transplant] wrote {a.dst}")
    print(f"[transplant] NEXT: resume with --reset-opt-layers {a.layers} -- the donated "
          f"layer's saved moments describe the weights it replaced")


if __name__ == "__main__":
    main()
