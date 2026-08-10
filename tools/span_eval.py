"""Evaluate a checkpoint under BOTH masking schemes.

The live v2 run trains on geometric spans but evaluates on scattered masking, so its
logged masked_acc stays comparable with every earlier run -- at the cost of never showing
how it is doing on the task it is actually optimising. This reports both, on the same
data with the same seed, so the pair can be read together.

Runs on CPU by default so it never contends with training for VRAM. Reading last.pt
while training is writing it is safe: checkpoints are written with os.replace, which
is atomic.

    python tools/span_eval.py --ckpt checkpoints/ckpt_v2/last.pt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch
from torch.utils.data import DataLoader

from mini_enc_transformer.data.dataset import MixtureDataset, PackedMemmapDataset
from mini_enc_transformer.model.mlm import BertForMaskedLM as BertV1
from mini_enc_transformer.model_v2.mlm import BertForMaskedLM as BertV2, IGNORE_INDEX, mask_tokens
from mini_enc_transformer.training.pretrain import build_tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ckpt_v2/last.pt")
    # Must match the training run's --data-name EXACTLY, limits included: the val split
    # is the tail 1% of the *limited* range, and the mixture weights set how much each
    # corpus contributes. Getting either wrong measures a different val set -- which it
    # did, reading 0.5878 where the run logged 0.5403.
    p.add_argument("--data-name",
                   default="cosmopedia:403451249:403451249,"
                           "ultrafineweb_en:256832885:256832885,"
                           "tinystories:73377468:73377468")
    p.add_argument("--batches", type=int, default=15)
    p.add_argument("--baseline", default="checkpoints/ckpt3/last.pt",
                   help="scored on the SAME val set and masks; '' to skip")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


@torch.no_grad()
def run(model, loader, ids, dev, batches, **kw):
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


def load_any(path, ids):
    """Build whichever architecture the checkpoint actually contains.

    ckpt3 is 4 layers / d_ff 3072 / ReLU (model), v2 is 6 / 1792 / GELU (model_v2).
    Both are read off the tensor shapes so this cannot silently load the wrong one.
    """
    sd = torch.load(path, map_location="cpu")["model"]
    n_layers = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("encoder.encoder_blocks."))
    d_ff = sd["encoder.encoder_blocks.0.feed_forward.l1"].shape[1]
    cls = BertV1 if (n_layers == 4 and d_ff == 3072) else BertV2
    kw = dict(vocab_size=ids["vocab_size"], d_model=768, d_k=64, d_v=64, n_heads=4,
              n_layers=n_layers, pad_id=ids["pad_id"], d_embed=128)
    m = cls(**kw) if cls is BertV1 else cls(d_ff=d_ff, **kw)
    m.load_state_dict(sd)
    m.eval()
    act = "ReLU" if cls is BertV1 else "GELU"
    return m, f"{n_layers}L d_ff={d_ff} {act}"


def main():
    a = parse_args()
    tk, ids = build_tokenizer("allenai/OLMo-1B-hf")
    dev = torch.device(a.device)

    specs = []
    for part in [x.strip() for x in a.data_name.split(",") if x.strip()]:
        bits = part.split(":")
        specs.append((bits[0],
                      float(bits[1]) if len(bits) > 1 and bits[1] else 1.0,
                      int(float(bits[2])) if len(bits) > 2 and bits[2] else None))
    dss = [PackedMemmapDataset("data", n, 128, "val", limit_tokens=l) for n, _, l in specs]
    val = dss[0] if len(dss) == 1 else MixtureDataset(dss, [w for _, w, _ in specs])
    loader = DataLoader(val, batch_size=16, shuffle=False)

    targets = [(a.ckpt, "v2")]
    if a.baseline and os.path.exists(a.baseline) and a.baseline != a.ckpt:
        # Label from the path, not hardcoded: --baseline is often NOT ckpt3 (a pre-anneal
        # checkpoint, another branch), and a wrong label silently misreads the table.
        base = os.path.basename(a.baseline)
        label = (os.path.basename(os.path.dirname(a.baseline)) if base == "last.pt"
                 else base.replace(".pt", ""))
        targets.append((a.baseline, label))

    out = []
    for path, label in targets:
        m, desc = load_any(path, ids)
        m.to(dev)
        step = torch.load(path, map_location="cpu").get("step", "?")
        # identical seed + unshuffled loader => both models see the same masked inputs
        sc_l, sc_a = run(m, loader, ids, dev, a.batches, span_min=1, span_max=1)
        sp_l, sp_a = run(m, loader, ids, dev, a.batches, span_min=1, span_max=10,
                         span_dist="geometric", geom_p=0.2)
        out.append((label, step, desc, sc_l, sc_a, sp_l, sp_a))
        del m

    hdr = (f"{'model':<7} {'step':>6} {'architecture':<18} "
           f"{'scat_loss':>10} {'scat_acc':>9} {'span_loss':>10} {'span_acc':>9} {'gap':>7}")
    print(hdr)
    print("-" * len(hdr))
    for label, step, desc, sc_l, sc_a, sp_l, sp_a in out:
        print(f"{label:<7} {str(step):>6} {desc:<18} "
              f"{sc_l:10.4f} {sc_a:9.4f} {sp_l:10.4f} {sp_a:9.4f} {sp_l-sc_l:+7.3f}")
    if len(out) == 2:
        v, c = out[0], out[1]
        print("-" * len(hdr))
        print(f"{'delta':<7} {'':>6} {'v2 - ckpt3':<18} "
              f"{v[3]-c[3]:+10.4f} {v[4]-c[4]:+9.4f} {v[5]-c[5]:+10.4f} {v[6]-c[6]:+9.4f} "
              f"{(v[5]-v[3])-(c[5]-c[3]):+7.3f}")


if __name__ == "__main__":
    main()
