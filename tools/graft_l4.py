"""Function-preserving re-initialisation of a dead layer, in place.

Layer 4 was stacked from L2 and never became a layer. Four independent measures agree
it is sitting at its initialisation: 97% of its FFN never driven positive, utilisation
0.07, attention entropy 3.6-4.0 against a 4.85 uniform ceiling, and centred head
similarity -0.23 against a measured random-init null of -0.20.

The graft:

    W_q, W_k, W_v   fresh N(0, 0.02)
    W_O             ZERO
    l1              fresh N(0, 0.02),  b1 zero
    l2              ZERO,              b2 zero
    layer_norm_1/2  UNTOUCHED

Zeroing the two output projections is the point. The rest of the stack has converged
around whatever L4 currently emits, so a random graft would inject noise into a
converged model -- that is what the first attempt did, and it cost ~4,000 steps of
recovery. With both output projections zero the layer emits nothing, so it cannot
disrupt anything while it learns.

CAVEAT, measured rather than assumed: this model is post-LN, so the block computes
ln2(ln1(x)) rather than the identity when its outputs are zero. Zero-init is only an
exact identity in a pre-LN architecture. It is close here only because L4 already
contributes almost nothing -- so this script MEASURES the loss on both sides of the
graft and refuses to write if the jump exceeds --max-delta.

Gradient flow at step 0: W_O and l2 receive gradient immediately (their inputs are
non-zero), but W_q/W_k/W_v and l1 receive none until W_O and l2 lift off zero, since
their path to the loss is multiplied by them. The layer is therefore inert for a few
steps by construction. That is also why these matrices must sit in a weight-decay-free
group -- decay competing with a weak early gradient is what keeps a zeroed matrix small.

    python tools/graft_l4.py --ckpt checkpoints/ckpt_v2/last.pt --layers 4
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
    p.add_argument("--backup", default=None,
                   help="copy the checkpoint here before overwriting (in-place only)")
    p.add_argument("--layers", default="4", help="comma-separated layer indices to graft")
    p.add_argument("--std", type=float, default=0.02, help="init std, matching the model")
    p.add_argument("--seqs", type=int, default=64, help="sequences for the continuity check")
    p.add_argument("--max-delta", type=float, default=0.15,
                   help="refuse to write if val loss jumps more than this many nats")
    p.add_argument("--corpora", default="cosmopedia,ultrafineweb_en,tinystories")
    p.add_argument("--force", action="store_true", help="write even if the delta is large")
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
def loss_on(model, masked, labels):
    tot, n = 0.0, 0
    for i in range(0, masked.shape[0], 16):
        out = model(masked[i:i + 16], labels[i:i + 16])
        tot += float(out["loss"]) * masked[i:i + 16].shape[0]
        n += masked[i:i + 16].shape[0]
    return tot / max(1, n)


def main():
    a = parse_args()
    layers = [int(x) for x in a.layers.split(",") if x.strip()]
    out_path = a.out or a.ckpt
    tk, ids = build_tokenizer("allenai/OLMo-1B-hf")

    ck = torch.load(a.ckpt, map_location="cpu")
    sd = ck["model"]
    n_layers = 1 + max(int(k.split(".")[2]) for k in sd
                       if k.startswith("encoder.encoder_blocks."))
    d_ff = sd["encoder.encoder_blocks.0.feed_forward.l1"].shape[1]
    model = BertV2(vocab_size=ids["vocab_size"], d_model=768, d_k=64, d_v=64, n_heads=4,
                   n_layers=n_layers, d_ff=d_ff, pad_id=ids["pad_id"], d_embed=128)
    model.load_state_dict(sd)
    model.eval()
    print(f"[graft] {a.ckpt}  step {ck.get('step', '?')}  {n_layers}L d_ff={d_ff}", flush=True)

    # identical masked batch on both sides of the graft, so the delta is the graft alone
    torch.manual_seed(1234)
    block = fixed_batch([s.strip() for s in a.corpora.split(",") if s.strip()], a.seqs)
    masked, labels = mask_tokens(block, ids["mask_id"], ids["vocab_size"],
                                 special_token_ids=ids["special_ids"],
                                 pad_token_id=ids["pad_id"], mlm_probability=0.15,
                                 span_min=1, span_max=1)

    before = loss_on(model, masked, labels)
    print(f"[graft] loss before      {before:.4f}", flush=True)

    torch.manual_seed(0)
    with torch.no_grad():
        for i in layers:
            blk = model.encoder.encoder_blocks[i]
            m, ff = blk.mha, blk.feed_forward
            for W in (m.W_q, m.W_k, m.W_v):
                W.normal_(0.0, a.std)
            m.W_O.zero_()
            ff.l1.normal_(0.0, a.std)
            ff.b1.zero_()
            ff.l2.zero_()
            ff.b2.zero_()
            print(f"[graft] L{i}: W_q/W_k/W_v ~ N(0,{a.std}), l1 ~ N(0,{a.std}); "
                  f"W_O and l2 ZEROED; layer norms kept", flush=True)

    after = loss_on(model, masked, labels)
    delta = after - before
    print(f"[graft] loss after       {after:.4f}   delta {delta:+.4f} nats", flush=True)

    if abs(delta) > a.max_delta and not a.force:
        print(f"[graft] ABORT: |delta| {abs(delta):.4f} > --max-delta {a.max_delta}. "
              f"Nothing written. Post-LN means the graft is not exactly "
              f"function-preserving; re-run with --force to accept.", flush=True)
        sys.exit(1)

    if a.backup and out_path == a.ckpt:
        shutil.copy2(a.ckpt, a.backup)
        print(f"[graft] backed up original -> {a.backup}", flush=True)

    ck["model"] = model.state_dict()
    ck["graft"] = {"layers": layers, "std": a.std, "loss_before": round(before, 4),
                   "loss_after": round(after, 4), "delta": round(delta, 4),
                   "zeroed": ["mha.W_O", "feed_forward.l2", "feed_forward.b2"],
                   "at_step": ck.get("step")}
    # the optimiser moments for these params describe weights that no longer exist;
    # the trainer clears them via --reset-opt-layers on resume
    tmp = out_path + ".tmp"
    torch.save(ck, tmp)
    os.replace(tmp, out_path)
    print(f"[graft] wrote {out_path}", flush=True)
    print(f"[graft] NEXT: resume with --reset-opt-layers {a.layers} so the stale Adam "
          f"moments for these params are dropped", flush=True)


if __name__ == "__main__":
    main()
