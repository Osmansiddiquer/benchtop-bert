"""How much harder is span masking than scattered, on identical data?

This does NOT answer "does training on spans help" -- that needs matched training runs.
It isolates the *task difficulty* difference by evaluating one fixed model under several
masking schemes with the same token budget, same data, same seed. That number is what
makes train/val curves incomparable across schemes, so it is worth knowing exactly.

    python tools/masking_difficulty.py --ckpt checkpoints/ckpt3/last.pt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch
from torch.utils.data import DataLoader

from mini_enc_transformer.data.dataset import MixtureDataset, PackedMemmapDataset
from mini_enc_transformer.model.mlm import IGNORE_INDEX, mask_tokens
from mini_enc_transformer.training.pretrain import build_tokenizer, build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ckpt3/last.pt")
    p.add_argument("--corpora", default="ultrafineweb_en,tinystories,imdb")
    p.add_argument("--batches", type=int, default=40)
    p.add_argument("--mlm-probs", default="0.15,0.25")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


class C:
    d_model, d_k, d_v, n_heads, n_layers, d_embed = 768, 64, 64, 4, 4, 128


SCHEMES = [
    ("scattered",        dict(span_min=1,  span_max=1)),
    ("uniform 2-4",      dict(span_min=2,  span_max=4,  span_dist="uniform")),
    ("geometric p=.2",   dict(span_min=1,  span_max=10, span_dist="geometric", geom_p=0.2)),
    ("geometric p=.1",   dict(span_min=1,  span_max=10, span_dist="geometric", geom_p=0.1)),
]


@torch.no_grad()
def score(model, loader, ids, dev, mlm_prob, kw, batches):
    """Same seed per scheme, so every scheme sees the same underlying text."""
    g = torch.Generator().manual_seed(1234)
    tot, correct, scored, n, runs = 0.0, 0, 0, 0, []
    for block in loader:
        if n >= batches:
            break
        n += 1
        masked, labels = mask_tokens(block, ids["mask_id"], ids["vocab_size"],
                                     special_token_ids=ids["special_ids"],
                                     pad_token_id=ids["pad_id"], mlm_probability=mlm_prob,
                                     generator=g, **kw)
        sel = labels != IGNORE_INDEX
        if n <= 4:                                  # measure realised span lengths
            for row in sel:
                c = 0
                for v in row.tolist():
                    if v:
                        c += 1
                    elif c:
                        runs.append(c); c = 0
                if c:
                    runs.append(c)
        out = model(masked.to(dev), labels.to(dev))
        tot += out["loss"].item()
        s = sel.to(dev)
        correct += (out["logits"].argmax(-1)[s] == labels.to(dev)[s]).sum().item()
        scored += s.sum().item()
    mean_run = sum(runs) / max(1, len(runs))
    return tot / max(1, n), correct / max(1, scored), mean_run


def main():
    a = parse_args()
    tk, ids = build_tokenizer("allenai/OLMo-1B-hf")
    model = build_model(C(), ids)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])
    dev = torch.device(a.device)
    model.to(dev).eval()

    names = [s.strip() for s in a.corpora.split(",") if s.strip()]
    val = MixtureDataset([PackedMemmapDataset("data", n, 128, "val") for n in names],
                         [1.0] * len(names))
    loader = DataLoader(val, batch_size=16, shuffle=False)

    print(f"[difficulty] {a.ckpt} -- one fixed model, identical data and seed\n")
    for mp in [float(x) for x in a.mlm_probs.split(",")]:
        print(f"  mlm_prob = {mp}")
        print(f"    {'scheme':17} {'mean span':>10} {'mlm_loss':>9} {'masked_acc':>11} {'vs scattered':>13}")
        base = None
        for name, kw in SCHEMES:
            loss, acc, run = score(model, loader, ids, dev, mp, kw, a.batches)
            base = base if base is not None else loss
            print(f"    {name:17} {run:10.2f} {loss:9.4f} {acc:11.4f} "
                  f"{loss-base:+13.4f}")
        print()
    print("  A model trained on one scheme and evaluated on another is comparing")
    print("  different tasks; this is the size of that offset.")


if __name__ == "__main__":
    main()
