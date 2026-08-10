"""Low-rank factorisation of the FFN matrices -- shrink parameters without deleting units.

The alternative to culling d_ff. Culling deletes hidden units, i.e. removes key-value
memory slots; if early-layer FFNs are sparse token/n-gram detectors, those slots are the
capacity, not slack. Factorisation instead compresses the *write subspace*: replace
W (d_in x d_out) with A(d_in x k) @ B(k x d_out), keeping all d_ff detectors alive.

Parameters go from d_in*d_out to k*(d_in + d_out), so for the 768x3072 pair the
break-even is k < 614. Below that it is a strict win over the original matrix.

This measures the QUALITY cost by substituting the rank-k reconstruction and evaluating
MLM loss; savings are exact arithmetic. Run before committing to any architecture.

    python tools/ffn_factorize.py --ranks 256,384,512 --target both
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy

import torch
from torch.utils.data import DataLoader

from mini_enc_transformer.data.dataset import MixtureDataset, PackedMemmapDataset
from mini_enc_transformer.training.pretrain import build_tokenizer, build_model, evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ckpt3/last.pt")
    p.add_argument("--ranks", default="256,384,512,614")
    p.add_argument("--target", choices=["l1", "l2", "both"], default="both")
    p.add_argument("--corpora", default="ultrafineweb_en,tinystories,imdb")
    p.add_argument("--eval-batches", type=int, default=30)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


class C:
    d_model, d_k, d_v, n_heads, n_layers, d_embed = 768, 64, 64, 4, 4, 128


def low_rank(W, k):
    """Best rank-k approximation (Eckart-Young). Returned as the reconstruction so the
    module shape is unchanged -- quality is what we are measuring here, not speed."""
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    return (U[:, :k] * S[:k]) @ Vh[:k], float((S[:k] ** 2).sum() / (S ** 2).sum())


def params_for(d_ff, k, target):
    """Per-layer FFN parameters after factorising the chosen matrices at rank k."""
    full = 768 * d_ff
    fact = k * (768 + d_ff)
    l1 = fact if target in ("l1", "both") else full
    l2 = fact if target in ("l2", "both") else full
    return l1 + l2 + d_ff + 768          # + biases


def main():
    a = parse_args()
    tk, ids = build_tokenizer("allenai/OLMo-1B-hf")
    names = [s.strip() for s in a.corpora.split(",") if s.strip()]
    val = MixtureDataset([PackedMemmapDataset("data", n, 128, "val") for n in names],
                         [1.0] * len(names))
    loader = DataLoader(val, batch_size=16, shuffle=False)
    dev = torch.device(a.device)

    base = build_model(C(), ids)
    base.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])
    base.to(dev).eval()
    b_loss, b_acc = evaluate(base, loader, ids, dev, max_batches=a.eval_batches)
    print(f"[factorise] baseline  mlm_loss {b_loss:.4f}  masked_acc {b_acc:.4f}\n")

    print(f"target={a.target}   break-even rank = 614 (below this beats the full matrix)")
    print(f"{'rank':>5} {'energy kept':>12} {'mlm_loss':>9} {'d_loss':>8} {'acc':>7} "
          f"{'ffn/layer':>10} {'6-layer total':>14}")
    for k in [int(x) for x in a.ranks.split(",")]:
        m = copy.deepcopy(base)
        energies = []
        with torch.no_grad():
            for blk in m.encoder.encoder_blocks:
                ff = blk.feed_forward
                if a.target in ("l1", "both"):
                    W, e = low_rank(ff.l1.data, k); ff.l1.data.copy_(W); energies.append(e)
                if a.target in ("l2", "both"):
                    W, e = low_rank(ff.l2.data, k); ff.l2.data.copy_(W); energies.append(e)
        loss, acc = evaluate(m, loader, ids, dev, max_batches=a.eval_batches)
        per_layer = params_for(3072, k, a.target)
        total = 6 * (786_432 + per_layer) + 6_534_000 + 149_000     # 6 layers + emb + head
        print(f"{k:>5} {100*sum(energies)/len(energies):11.1f}% {loss:9.4f} "
              f"{loss-b_loss:+8.4f} {acc:7.4f} {per_layer/1e6:9.3f}M {total/1e6:13.2f}M")
        del m

    print(f"\ncurrent 4-layer d_ff 3072 unfactorised: 28.73M")
    print("6-layer totals above keep ALL 3072 detectors per layer.")


if __name__ == "__main__":
    main()
