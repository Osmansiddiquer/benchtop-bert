"""Are the 'dead' FFN neurons actually dead, or rare-token detectors?

The earlier measurement called a unit dead if it never fired across 8,192 tokens. That
sample cannot distinguish a broken unit from a detector for a token appearing once per
100k tokens -- and Voita et al. find exactly such sparse token/n-gram detectors dominate
early FFN layers, becoming *more* common with scale. If that is what layer 0's 2,755
'dead' units are, pruning them destroys a sparse lookup table rather than removing slack.

Probes only the corpora ckpt3 actually trained on (UltraFineWeb, TinyStories, IMDB);
Cosmopedia is out-of-distribution for it and would confound the firing rates.

    python tools/dead_neuron_check.py --tokens 2000000
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import numpy as np
import torch

from mini_enc_transformer.training.pretrain import build_tokenizer, build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ckpt3/last.pt")
    p.add_argument("--corpora", default="ultrafineweb_en,tinystories,imdb",
                   help="ckpt3's training corpora only")
    p.add_argument("--tokens", type=int, default=2_000_000)
    return p.parse_args()


class C:
    d_model, d_k, d_v, n_heads, n_layers, d_embed = 768, 64, 64, 4, 4, 128


def main():
    a = parse_args()
    tk, ids = build_tokenizer("allenai/OLMo-1B-hf")
    model = build_model(C(), ids)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])
    model.eval()

    names = [s.strip() for s in a.corpora.split(",") if s.strip()]
    per = a.tokens // len(names)
    fired = [torch.zeros(3072) for _ in range(C.n_layers)]
    n_tok = 0
    for nm in names:
        man = json.load(open(f"data/{nm}.manifest.json"))
        mm = np.memmap(f"data/{nm}.bin", dtype=np.uint16, mode="r",
                       shape=(man["target_tokens"],))
        take = min(per, man["tokens_written"] - 1)
        take -= take % 128
        seqs = torch.from_numpy(np.asarray(mm[:take]).astype(np.int64)).view(-1, 128)
        with torch.no_grad():
            for chunk in seqs.split(64):
                x = model.encoder.pe(model.encoder.embedding(chunk))
                for i, blk in enumerate(model.encoder.encoder_blocks):
                    h = blk.layer_norm_1(blk.mha(x, False) + x)
                    ff = blk.feed_forward
                    act = ff.relu_1(h @ ff.l1 + ff.b1).reshape(-1, 3072)
                    fired[i] += (act > 0).float().sum(0)
                    x = blk.layer_norm_2(ff(h) + h)
                n_tok += chunk.numel()
        print(f"  probed {nm}: running total {n_tok:,} tokens", flush=True)

    print(f"\nfiring rates over {n_tok:,} tokens (earlier claim used 8,192)")
    print(f"{'layer':6} {'never':>8} {'<1/100k':>9} {'<1/10k':>8} {'<1/1k':>8} "
          f"{'<1%':>7} {'>=1%':>7}")
    for i, f in enumerate(fired):
        r = f / n_tok
        print(f"{i:<6} {int((f == 0).sum()):8d} "
              f"{int(((r > 0) & (r < 1e-5)).sum()):9d} "
              f"{int(((r >= 1e-5) & (r < 1e-4)).sum()):8d} "
              f"{int(((r >= 1e-4) & (r < 1e-3)).sum()):8d} "
              f"{int(((r >= 1e-3) & (r < 1e-2)).sum()):7d} "
              f"{int((r >= 1e-2).sum()):7d}")
    print("\n'never' at this sample size is far stronger evidence of a truly dead unit.")
    print("Units in the sparse columns are candidate rare-token detectors -- pruning")
    print("those removes memory slots, not redundancy.")


if __name__ == "__main__":
    main()
