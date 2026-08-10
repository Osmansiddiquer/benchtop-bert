"""Measure what it costs to append a layer to a trained post-LN stack.

This reproduces the main result of `paper/report.pdf` (Section 4). The short version:

    A zero-initialised graft IS exactly function-preserving in post-LN -- but only if you
    append it at the TOP of the stack, and only if you RESET the new block's LayerNorm
    affines to (gamma=1, beta=0) instead of copying them from the donor layer.

The folklore says zero-init grafting only works in pre-LN, where a block is `x + f(LN(x))`
and `f = 0` is trivially the identity. In post-LN the block is

    h   = LN1(MHA(x) + x)
    out = LN2(FFN(h) + h)

so zeroing W_O and the FFN output leaves `out = LN2(LN1(x))` -- two extra LayerNorms, which
is NOT the identity map. It is nonetheless function-preserving *in the network*, because
LayerNorm is invariant to per-token affine rescaling,

    LN(a*x + b) = LN(x)     for scalar a > 0, b

and LN(x) is exactly such a rescaling of x. So whenever the next consumer of the stream is
another LayerNorm -- which at the top of the stack it is, namely encoder.final_norm -- the
graft is invisible. Mid-stack it is not: the following block's MHA reads the residual stream
directly (attention logits scale as the square of its magnitude), and the graft has just
stripped the previous block's learned output scale.

Measured on ckpt3, appending one block (scattered loss, fixed held-out mix):

    ckpt3 4L/3072 ReLU (source)                         2.7089
    GELU swap only, still 4L  <- the reference          2.9776   +0.000
    W_O=0, FFN_out=0, LN reset to (1,0), appended       2.9776   +0.000   max|dlogit| 0.0014
    W_O=0, FFN_out=0, LN copied from donor              7.1016   +4.124   max|dlogit| 37.6
    only FFN_out=0 (attention live), LN copied          7.7290   +4.751
    full copy-stack (progressive stacking)              8.5488   +5.571

    inserted at position 0                              3.0646   +0.087
    inserted at position 2 (mid-stack)                  3.2021   +0.225
    appended at position 4 (top)                        2.9776   +0.000

The dominant error term is COPYING THE DONOR'S LAYERNORM AFFINES (+4.12 of +5.57), which is
the opposite of the stacking instinct: gamma and beta encode what scale the next layer
expects, so a new block inherits the wrong expectation.

    python tools/graft_ladder.py --device cuda
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy
import json

import torch
from torch.utils.data import DataLoader

from mini_enc_transformer.data.dataset import MixtureDataset, PackedMemmapDataset
from mini_enc_transformer.model.mlm import BertForMaskedLM as BertV1
from mini_enc_transformer.model_v2.mlm import (BertForMaskedLM as BertV2, IGNORE_INDEX,
                                               mask_tokens)
from mini_enc_transformer.training.pretrain import build_tokenizer

# Same limits the v2 run trained under; the val split is the tail 1% of the LIMITED range,
# so getting these wrong measures a different held-out set entirely.
VAL_MIX = [("cosmopedia", 403451249), ("ultrafineweb_en", 256832885),
           ("tinystories", 73377468)]

BLOCKS = "encoder.encoder_blocks."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ckpt3/last.pt",
                   help="a trained 4-layer v1 checkpoint to graft onto")
    p.add_argument("--builder", default="tools/build_v2_from_ckpt3.py")
    p.add_argument("--work-dir", default=None,
                   help="where to write the variant checkpoints (default: a temp dir)")
    p.add_argument("--batches", type=int, default=15)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="results/graft_ladder.json")
    return p.parse_args()


def build_loader():
    dss = [PackedMemmapDataset("data", n, 128, "val", limit_tokens=lim) for n, lim in VAL_MIX]
    # weights proportional to each corpus's token budget -> the 55/35/10 Phase-A mixture
    mix = MixtureDataset(dss, [float(lim) for _, lim in VAL_MIX])
    return DataLoader(mix, batch_size=16, shuffle=False)


def load(path, ids, dev, force_v2=True):
    sd = torch.load(path, map_location="cpu")["model"]
    n_layers = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith(BLOCKS))
    d_ff = sd[f"{BLOCKS}0.feed_forward.l1"].shape[1]
    cls = BertV2 if force_v2 else (BertV1 if (n_layers == 4 and d_ff == 3072) else BertV2)
    kw = dict(vocab_size=ids["vocab_size"], d_model=768, d_k=64, d_v=64, n_heads=4,
              n_layers=n_layers, pad_id=ids["pad_id"], d_embed=128)
    m = cls(**kw) if cls is BertV1 else cls(d_ff=d_ff, **kw)
    m.load_state_dict(sd)
    return m.eval().to(dev), n_layers, d_ff


@torch.no_grad()
def score(model, loader, ids, dev, batches, **kw):
    # identical seed + unshuffled loader => every variant sees byte-identical masked inputs
    g = torch.Generator().manual_seed(1234)
    tot, correct, scored, n = 0.0, 0, 0, 0
    for block in loader:
        if n >= batches:
            break
        n += 1
        masked, labels = mask_tokens(block, ids["mask_id"], ids["vocab_size"],
                                     special_token_ids=ids["special_ids"],
                                     pad_token_id=ids["pad_id"], generator=g, **kw)
        out = model(masked.to(dev), labels.to(dev))
        tot += out["loss"].item()
        sel = (labels != IGNORE_INDEX).to(dev)
        correct += (out["logits"].argmax(-1)[sel] == labels.to(dev)[sel]).sum().item()
        scored += sel.sum().item()
    return tot / max(1, n), correct / max(1, scored)


def edit_block(ck, idx, *, zero_wo, zero_ffn, neutral_ln):
    """Return a copy of `ck` with block `idx` turned into a graft of the requested kind."""
    out = copy.deepcopy(ck)
    sd, pre = out["model"], f"{BLOCKS}{idx}."
    if zero_wo:
        sd[pre + "mha.W_O"] = torch.zeros_like(sd[pre + "mha.W_O"])
    if zero_ffn:
        for k in ("feed_forward.l2", "feed_forward.b2"):
            sd[pre + k] = torch.zeros_like(sd[pre + k])
    if neutral_ln:
        for k in list(sd):
            if k.startswith(pre) and "layer_norm" in k:
                sd[k] = (torch.ones_like(sd[k]) if k.endswith("gamma")
                         else torch.zeros_like(sd[k]))
    return out


def insert_block(ck4, pos, n_layers=5):
    """Build an `n_layers`-deep state dict by inserting a fully-neutralised graft at `pos`.

    The donor is the nearest trained block. Everything outside encoder_blocks.* is copied.
    """
    src = ck4["model"]
    keep = {k: v.clone() for k, v in src.items() if not k.startswith(BLOCKS)}

    def block(i):
        p = f"{BLOCKS}{i}."
        return {k[len(p):]: v for k, v in src.items() if k.startswith(p)}

    donor, graft = block(min(pos, n_layers - 2)), {}
    for k, v in donor.items():
        if "layer_norm" in k:
            graft[k] = torch.ones_like(v) if k.endswith("gamma") else torch.zeros_like(v)
        elif k == "mha.W_O" or k in ("feed_forward.l2", "feed_forward.b2"):
            graft[k] = torch.zeros_like(v)
        else:
            graft[k] = v.clone()

    old = 0
    for i in range(n_layers):
        blk = graft if i == pos else block(old)
        if i != pos:
            old += 1
        for k, v in blk.items():
            keep[f"{BLOCKS}{i}.{k}"] = v.clone()
    out = dict(ck4)
    out["model"] = keep
    return out


def main():
    a = parse_args()
    work = a.work_dir or os.path.join(os.environ.get("TMPDIR", "/tmp"), "graft_ladder")
    os.makedirs(work, exist_ok=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    tk, ids = build_tokenizer("allenai/OLMo-1B-hf")
    dev = torch.device(a.device)
    loader = build_loader()

    # ---- the two builder outputs we need -------------------------------------------
    # 4L/3072 GELU: ckpt3's weights under the v2 block, i.e. the activation swap alone.
    # 5L/3072 with the FFN output zeroed: the starting point every variant is edited from.
    ref4 = f"{work}/ref_4L_gelu.pt"
    base5 = f"{work}/base_5L.pt"
    for out_path, extra in ((ref4, "--n-layers 4"),
                            (base5, "--n-layers 5 --new-layer-init zero")):
        if not os.path.exists(out_path):
            cmd = (f"python {a.builder} --ckpt {a.ckpt} --d-ff 3072 {extra} --out {out_path}")
            print(f"[build] {cmd}")
            if os.system(cmd) != 0:
                sys.exit(f"builder failed: {cmd}")
    copy5 = f"{work}/copy_5L.pt"
    if not os.path.exists(copy5):
        os.system(f"python {a.builder} --ckpt {a.ckpt} --d-ff 3072 --n-layers 5 "
                  f"--new-layer-init stack --out {copy5}")

    ck5 = torch.load(base5, map_location="cpu")
    ck4 = torch.load(ref4, map_location="cpu")

    variants = [("ckpt3 4L/3072 ReLU (source)", a.ckpt, False),
                ("GELU swap only, 4L  [REFERENCE]", ref4, True)]

    # the ladder: progressively more of the new block neutralised
    for name, kw in [("+1 block, copy-stacked", None),
                     ("+1 block, FFN_out=0, LN copied", dict(zero_wo=False, zero_ffn=True,
                                                             neutral_ln=False)),
                     ("+1 block, W_O=0 FFN_out=0, LN copied", dict(zero_wo=True, zero_ffn=True,
                                                                   neutral_ln=False)),
                     ("+1 block, W_O=0 FFN_out=0, LN reset", dict(zero_wo=True, zero_ffn=True,
                                                                  neutral_ln=True))]:
        if kw is None:
            variants.append((name, copy5, True))
            continue
        p = f"{work}/lad_{len(variants)}.pt"
        torch.save(edit_block(ck5, 4, **kw), p)
        variants.append((name, p, True))

    # where the graft goes
    for pos in (0, 2, 4):
        p = f"{work}/ins{pos}.pt"
        torch.save(insert_block(ck4, pos), p)
        variants.append((f"fully-neutral graft inserted at position {pos}", p, True))

    rows, ref = [], None
    print(f"\n{'variant':44} {'arch':>10} {'scat_loss':>10} {'delta':>8} {'scat_acc':>9}")
    print("-" * 86)
    for name, path, as_v2 in variants:
        m, n_layers, d_ff = load(path, ids, dev, force_v2=as_v2)
        loss, acc = score(m, loader, ids, dev, a.batches, span_min=1, span_max=1)
        if "REFERENCE" in name:
            ref = loss
        d = "" if ref is None else f"{loss - ref:+8.3f}"
        print(f"{name:44} {f'{n_layers}L/{d_ff}':>10} {loss:10.4f} {d:>8} {acc:9.4f}")
        rows.append(dict(variant=name, n_layers=n_layers, d_ff=d_ff, scat_loss=loss,
                         scat_acc=acc, delta_vs_ref=None if ref is None else loss - ref))
        del m
        torch.cuda.empty_cache()

    # ---- the decisive check: logits, not loss --------------------------------------
    # A loss delta of "about zero" is a judgement call. Comparing max|dlogit| against the
    # mean logit magnitude is a pass/fail, and it is what actually establishes exactness.
    print("\nfunction-preservation check (random token ids, 4x128):")
    m4, _, _ = load(ref4, ids, dev)
    x = torch.randint(0, 50000, (4, 128), generator=torch.Generator().manual_seed(0)).to(dev)
    with torch.no_grad():
        base_logits = m4(x)["logits"]
    scale = base_logits.abs().mean().item()
    checks = {}
    for label, kw in [("LN reset to (1,0)", dict(zero_wo=True, zero_ffn=True, neutral_ln=True)),
                      ("LN copied from donor", dict(zero_wo=True, zero_ffn=True,
                                                    neutral_ln=False))]:
        p = f"{work}/chk.pt"
        torch.save(edit_block(ck5, 4, **kw), p)
        m5, _, _ = load(p, ids, dev)
        with torch.no_grad():
            dmax = (base_logits - m5(x)["logits"]).abs().max().item()
        checks[label] = dmax
        print(f"  {label:26} max|dlogit| = {dmax:9.5f}   ({dmax / scale:.2e} x mean logit)")
        del m5
    print(f"  {'mean |logit|':26} {scale:.4f}")

    with open(a.out, "w") as f:
        json.dump(dict(ckpt=a.ckpt, batches=a.batches, rows=rows,
                       logit_check=checks, mean_logit=scale), f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
