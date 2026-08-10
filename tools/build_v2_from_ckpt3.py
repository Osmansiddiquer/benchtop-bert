"""Build the v2 checkpoint (6 layers, d_ff 1792, GELU) from ckpt3's trained encoder.

Three transfers, each with a measured reason:

1. **FFN 3072 -> 1792 by structured pruning.** Column j of l1 and row j of l2 are one
   hidden unit; deleting the pair leaves every survivor bit-exact. Units are ranked by
   E[|a_j|] * ||l2[j,:]|| -- how much the unit actually contributes to the output.

   Importance is scored under **GELU, not ReLU**, because that is the activation the
   target model uses. Scoring under ReLU would discard units whose pre-activations are
   negative -- 2,755 of them in layer 0 -- but GELU gives those units a small non-zero
   output, so some deserve to live. Using the wrong activation to choose would throw
   away exactly the capacity the GELU switch is meant to recover.

2. **Layers 4-5 initialised by stacking**, copying the top two trained layers rather
   than random init (progressive stacking / bert2BERT). A duplicated layer starts from
   a useful function instead of noise.

3. Embedding, final norm and MLM head copy across unchanged.

    python tools/build_v2_from_ckpt3.py --out checkpoints/ckpt_v2_init.pt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from mini_enc_transformer.model_v2.mlm import BertForMaskedLM as BertV2
from mini_enc_transformer.training.pretrain import build_tokenizer, build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ckpt3/last.pt")
    p.add_argument("--out", default="checkpoints/ckpt_v2_init.pt")
    p.add_argument("--d-ff", type=int, default=1792)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--calib-seqs", type=int, default=256)
    p.add_argument("--new-layer-init", choices=["stack", "random", "zero"], default="stack",
                   help="how layers beyond the source depth start. stack = copy a trained "
                        "layer (measured to leave it a near-duplicate the model then "
                        "suppresses); random = fresh init; zero = copy but with the output "
                        "projection zeroed, so the layer is an exact no-op at step 0 and "
                        "the function is preserved")
    p.add_argument("--corpora", default="ultrafineweb_en,tinystories,imdb")
    return p.parse_args()


class C1:                       # source architecture
    d_model, d_k, d_v, n_heads, n_layers, d_embed = 768, 64, 64, 4, 4, 128


def main():
    a = parse_args()
    torch.manual_seed(0)
    tk, ids = build_tokenizer("allenai/OLMo-1B-hf")

    src = build_model(C1(), ids)
    src.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])
    src.eval()

    # ---- calibration text spanning every corpus ckpt3 saw ----------------------
    names = [s.strip() for s in a.corpora.split(",") if s.strip()]
    per = a.calib_seqs // len(names)
    chunks = []
    for nm in names:
        man = json.load(open(f"data/{nm}.manifest.json"))
        mm = np.memmap(f"data/{nm}.bin", dtype=np.uint16, mode="r",
                       shape=(man["target_tokens"],))
        off = min(5_000_000, max(0, man["tokens_written"] - per * 128 - 1))
        chunks.append(torch.from_numpy(
            np.asarray(mm[off:off + per * 128]).astype(np.int64)).view(-1, 128))
    calib = torch.cat(chunks)
    print(f"[v2] calibration {calib.shape[0]} seqs ({calib.numel():,} tokens)", flush=True)

    # ---- score hidden units under GELU ----------------------------------------
    n_src = C1.n_layers
    absmean = [torch.zeros(3072) for _ in range(n_src)]
    relu_dead = [torch.zeros(3072) for _ in range(n_src)]
    n = 0
    with torch.no_grad():
        for chunk in calib.split(32):
            x = src.encoder.pe(src.encoder.embedding(chunk))
            for i, blk in enumerate(src.encoder.encoder_blocks):
                h = blk.layer_norm_1(blk.mha(x, False) + x)
                ff = blk.feed_forward
                pre = (h @ ff.l1 + ff.b1).reshape(-1, 3072).float()
                absmean[i] += F.gelu(pre).abs().sum(0)          # GELU, the v2 activation
                relu_dead[i] = torch.maximum(relu_dead[i], pre.clamp_min(0).max(0).values)
                x = blk.layer_norm_2(ff(h) + h)
            n += chunk.shape[0] * chunk.shape[1]
    absmean = [m / n for m in absmean]

    dst = BertV2(vocab_size=ids["vocab_size"], d_model=768, d_k=64, d_v=64, n_heads=4,
                 n_layers=a.n_layers, d_ff=a.d_ff, pad_id=ids["pad_id"], d_embed=128)

    print(f"\n[v2] FFN 3072 -> {a.d_ff}, scored under GELU")
    print(f"{'layer':6} {'relu-dead':>10} {'relu-dead kept':>15} {'kept score %':>13}")
    keeps = []
    for i, blk in enumerate(src.encoder.encoder_blocks):
        ff = blk.feed_forward
        score = absmean[i] * ff.l2.detach().float().norm(dim=1)
        keep = torch.topk(score, a.d_ff).indices.sort().values
        keeps.append(keep)
        dead = (relu_dead[i] < 1e-6)
        print(f"{i:<6} {int(dead.sum()):10d} {int(dead[keep].sum()):15d} "
              f"{100*score[keep].sum()/score.sum():12.1f}%")

    # ---- transfer --------------------------------------------------------------
    with torch.no_grad():
        dst.encoder.embedding.load_state_dict(src.encoder.embedding.state_dict())
        dst.encoder.final_norm.load_state_dict(src.encoder.final_norm.state_dict())
        dst.mlm_head.dense.load_state_dict(src.mlm_head.dense.state_dict())
        dst.mlm_head.norm.load_state_dict(src.mlm_head.norm.state_dict())
        dst.mlm_head.decoder.bias.copy_(src.mlm_head.decoder.bias)

        for i in range(a.n_layers):
            # layers beyond the source depth reuse the top trained layers (stacking)
            src_i = i if i < n_src else n_src - (a.n_layers - i)
            s_blk, d_blk = src.encoder.encoder_blocks[src_i], dst.encoder.encoder_blocks[i]
            d_blk.mha.load_state_dict(s_blk.mha.state_dict())
            d_blk.layer_norm_1.load_state_dict(s_blk.layer_norm_1.state_dict())
            d_blk.layer_norm_2.load_state_dict(s_blk.layer_norm_2.state_dict())
            k = keeps[src_i]
            d_blk.feed_forward.l1.copy_(s_blk.feed_forward.l1[:, k])
            d_blk.feed_forward.b1.copy_(s_blk.feed_forward.b1[k])
            d_blk.feed_forward.l2.copy_(s_blk.feed_forward.l2[k])
            d_blk.feed_forward.b2.copy_(s_blk.feed_forward.b2)
            if i >= n_src:
                if a.new_layer_init == "random":
                    # leave the freshly-constructed random weights untouched
                    fresh = BertV2(vocab_size=ids["vocab_size"], d_model=768, d_k=64, d_v=64,
                                   n_heads=4, n_layers=1, d_ff=a.d_ff,
                                   pad_id=ids["pad_id"], d_embed=128)
                    d_blk.load_state_dict(fresh.encoder.encoder_blocks[0].state_dict())
                    print(f"[v2] layer {i} <- RANDOM init")
                elif a.new_layer_init == "zero":
                    d_blk.feed_forward.l2.zero_(); d_blk.feed_forward.b2.zero_()
                    print(f"[v2] layer {i} <- stacked from L{src_i}, output zeroed (no-op at init)")
                else:
                    print(f"[v2] layer {i} <- stacked from source layer {src_i}")

    # the MLM head decoder is tied to the word embedding, which was copied
    dst.mlm_head.tie_to(dst.encoder.embedding.weight)

    n_p = sum(p.numel() for p in dst.parameters())
    torch.save({"model": dst.state_dict(), "new_layer_init": a.new_layer_init,
                "arch": {"n_layers": a.n_layers, "d_ff": a.d_ff, "act": "gelu",
                         "d_model": 768, "d_embed": 128, "n_heads": 4, "d_k": 64, "d_v": 64},
                "built_from": a.ckpt, "step": 0}, a.out)
    print(f"\n[v2] {n_p/1e6:.3f}M params (source 28.733M) -> {a.out}")


if __name__ == "__main__":
    main()
