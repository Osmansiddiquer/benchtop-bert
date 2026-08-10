"""Structured FFN pruning: cull d_ff by deleting whole hidden units.

Column j of l1 produces hidden unit j; row j of l2 writes it back into the residual
stream. Delete them as a pair and every surviving unit computes exactly what it did
before -- nothing is approximated, unlike a factorisation W ~= UV.

Importance of a unit is how much it actually contributes to the output:

    score_j = E[a_j] * ||l2[j, :]||

Both terms matter: a unit that fires strongly but writes ~nothing is useless, and so is
one with a large output row that never fires.

Removing unit j drops E[a_j] * l2[j,:] from the output on average. Folding that into b2
preserves the mean output for free, which is what --bias-correct does.

Reports MLM loss/accuracy before and after so the cost is measured, not assumed.

    python tools/ffn_prune.py --d-ff 1792
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from mini_enc_transformer.data.dataset import MixtureDataset, PackedMemmapDataset
from mini_enc_transformer.training.pretrain import build_tokenizer, build_model, evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ckpt3/last.pt")
    p.add_argument("--d-ff", type=int, default=1792, help="target hidden width per layer")
    p.add_argument("--calib-seqs", type=int, default=256, help="sequences used to score units")
    p.add_argument("--eval-batches", type=int, default=30)
    p.add_argument("--corpora", default="ultrafineweb_en,tinystories,imdb")
    p.add_argument("--bias-correct", action="store_true", default=True)
    p.add_argument("--no-bias-correct", dest="bias_correct", action="store_false")
    p.add_argument("--out", default=None, help="save the pruned checkpoint here")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


class C:
    d_model, d_k, d_v, n_heads, n_layers, d_embed = 768, 64, 64, 4, 4, 128
    seq_len = 128


def collect_stats(model, ids_t, n_layers):
    """Mean |activation| per hidden unit, and the mean activation (for bias correction)."""
    absmean = [torch.zeros(3072) for _ in range(n_layers)]
    mean = [torch.zeros(3072) for _ in range(n_layers)]
    maxact = [torch.zeros(3072) for _ in range(n_layers)]
    n = 0
    with torch.no_grad():
        for chunk in ids_t.split(32):
            x = model.encoder.pe(model.encoder.embedding(chunk))
            for i, blk in enumerate(model.encoder.encoder_blocks):
                h = blk.layer_norm_1(blk.mha(x, False) + x)
                ff = blk.feed_forward
                a = ff.relu_1(h @ ff.l1 + ff.b1).reshape(-1, 3072).float()
                absmean[i] += a.abs().sum(0)
                mean[i] += a.sum(0)
                maxact[i] = torch.maximum(maxact[i], a.max(0).values)
                x = blk.layer_norm_2(ff(h) + h)
            n += chunk.shape[0] * chunk.shape[1]
    return [m / n for m in absmean], [m / n for m in mean], maxact, n


def main():
    a = parse_args()
    torch.manual_seed(0)
    tk, ids = build_tokenizer("allenai/OLMo-1B-hf")
    model = build_model(C(), ids)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])
    model.eval()

    # calibration data spanning every corpus the model was trained on
    names = [s.strip() for s in a.corpora.split(",") if s.strip()]
    chunks = []
    per = a.calib_seqs // len(names)
    for nm in names:
        man = json.load(open(f"data/{nm}.manifest.json"))
        mm = np.memmap(f"data/{nm}.bin", dtype=np.uint16, mode="r", shape=(man["target_tokens"],))
        off = min(5_000_000, max(0, man["tokens_written"] - per * 128 - 1))
        chunks.append(torch.from_numpy(
            np.asarray(mm[off:off + per * 128]).astype(np.int64)).view(-1, 128))
    calib = torch.cat(chunks)
    print(f"[prune] calibration: {calib.shape[0]} sequences ({calib.numel():,} tokens) "
          f"from {', '.join(names)}", flush=True)

    # held-out data for the before/after comparison, disjoint from calibration
    val = MixtureDataset([PackedMemmapDataset("data", n, 128, "val") for n in names],
                         [1.0] * len(names)) if len(names) > 1 else \
          PackedMemmapDataset("data", names[0], 128, "val")
    val_loader = DataLoader(val, batch_size=16, shuffle=False)
    dev = torch.device(a.device)
    model.to(dev)
    before = evaluate(model, val_loader, ids, dev, max_batches=a.eval_batches)
    print(f"[prune] BEFORE  mlm_loss {before[0]:.4f}  masked_acc {before[1]:.4f}", flush=True)

    absmean, mean, maxact, ntok = collect_stats(model, calib, C.n_layers)

    print(f"\n[prune] 3072 -> {a.d_ff} per layer, bias_correct={a.bias_correct}")
    print(f"{'layer':6} {'dead':>6} {'kept dead':>10} {'score kept':>12} {'score dropped':>14}")
    for i, blk in enumerate(model.encoder.encoder_blocks):
        ff = blk.feed_forward
        out_norm = ff.l2.detach().float().norm(dim=1)          # ||l2[j, :]||
        score = absmean[i] * out_norm
        keep = torch.topk(score, a.d_ff).indices.sort().values
        drop = torch.tensor(sorted(set(range(3072)) - set(keep.tolist())))
        dead = int((maxact[i] < 1e-6).sum())
        kept_dead = int((maxact[i][keep] < 1e-6).sum())
        print(f"{i:<6} {dead:6d} {kept_dead:10d} {score[keep].sum():12.2f} "
              f"{score[drop].sum():14.2f}")

        if a.bias_correct and len(drop):
            # mean output lost by the dropped units, folded into b2
            delta = (mean[i][drop].unsqueeze(0) @ ff.l2.detach().float()[drop]).squeeze(0)
            ff.b2.data += delta.to(ff.b2.dtype)
        ff.l1 = torch.nn.Parameter(ff.l1.detach()[:, keep].clone())
        ff.b1 = torch.nn.Parameter(ff.b1.detach()[keep].clone())
        ff.l2 = torch.nn.Parameter(ff.l2.detach()[keep].clone())

    after = evaluate(model, val_loader, ids, dev, max_batches=a.eval_batches)
    print(f"\n[prune] BEFORE  mlm_loss {before[0]:.4f}  masked_acc {before[1]:.4f}")
    print(f"[prune] AFTER   mlm_loss {after[0]:.4f}  masked_acc {after[1]:.4f}")
    print(f"[prune] COST    +{after[0]-before[0]:.4f} nats  {100*(after[1]-before[1]):+.2f} acc points")

    n_new = sum(p.numel() for p in model.parameters())
    print(f"\n[prune] params {28.733:.3f}M -> {n_new/1e6:.3f}M "
          f"({100*(1-n_new/28_733_000):.1f}% smaller)")
    if a.out:
        torch.save({"model": model.state_dict(), "d_ff": a.d_ff, "pruned_from": a.ckpt}, a.out)
        print(f"[prune] saved {a.out}")


if __name__ == "__main__":
    main()
