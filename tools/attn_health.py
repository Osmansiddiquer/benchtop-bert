"""Attention-side health, the analogue of ffn_slack.

Everything measured so far has been the FFN. Attention is the other half, and on this
model it is the unusual half: n_heads * d_k = 256 against d_model 768, so the block
writes a rank-<=256 update into a 768-dimensional stream and holds only ~11% of the
parameters (BERT-base gives attention ~33%).

Per head h in layer l:

    contrib_h = std_t(o_h) * ||W_O[h-block, :]||     how much the head moves the stream
    entropy_h = mean over queries of H(attn row)      what the head is doing:
                  ~log(T)  -> uniform, i.e. averaging, not selecting
                  ~0       -> fixated on a single position
    offset_h  = mean |argmax(attn) - query index|     positional vs content-driven

Layer level:

    out_rank  effective rank of the attention output, against the 256 ceiling. Near 256
              means the narrow width is genuinely binding; well below means it is not.
    head_sim  max pairwise cosine between heads' attention patterns, CENTRED across
              heads. Centring is not optional: raw cosine has a null of ~0.97 rather
              than 0, because an untrained model leaves every head at near-uniform
              attention and the shared 1/T floor dominates the cosine. Measured on a
              fresh BertV2: raw 0.95-0.99 per layer, centred -0.23..-0.07. An uncentred
              reading therefore cannot distinguish "duplicated heads" from "untrained
              heads" -- exactly the two cases this tool exists to tell apart.

              The null is NEGATIVE, not zero: four centred vectors sum to zero, so
              undifferentiated heads are mutually anti-correlated. A layer reading ~-0.2
              has not left its initialisation.

              It is a MAX over pairs, so it detects the existence of one duplicated
              pair, not overall diversity; 'dup' names the pair responsible.

Reports head_dead = heads whose contribution is <1% of the layer's p90, matching the
FFN slack convention so the two panels read the same way.

    python tools/attn_health.py --ckpt checkpoints/ckpt_v2/last.pt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from mini_enc_transformer.model.mlm import BertForMaskedLM as BertV1
from mini_enc_transformer.model_v2.mlm import BertForMaskedLM as BertV2
from mini_enc_transformer.training.pretrain import build_tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ckpt_v2/last.pt")
    p.add_argument("--corpora", default="cosmopedia,ultrafineweb_en,tinystories")
    p.add_argument("--seqs", type=int, default=12)
    p.add_argument("--history", default="logs/attn_health.jsonl")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def load_any(path, ids):
    sd = torch.load(path, map_location="cpu")["model"]
    n_layers = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("encoder.encoder_blocks."))
    d_ff = sd["encoder.encoder_blocks.0.feed_forward.l1"].shape[1]
    cls = BertV1 if (n_layers == 4 and d_ff == 3072) else BertV2
    kw = dict(vocab_size=ids["vocab_size"], d_model=768, d_k=64, d_v=64, n_heads=4,
              n_layers=n_layers, pad_id=ids["pad_id"], d_embed=128)
    m = cls(**kw) if cls is BertV1 else cls(d_ff=d_ff, **kw)
    m.load_state_dict(sd)
    m.eval()
    return m, n_layers, d_ff


def _lo(man, nm, per):
    lo, hi = val_offset(man, LIMITS.get(nm))
    need = per * 128
    return lo if hi - lo >= need else max(0, hi - need)


def val_offset(man, limit, seq_len=128, val_fraction=0.01):
    """First token of the HELD-OUT range, matching PackedMemmapDataset exactly.
    Hardcoded offsets silently read TRAINING data -- this tool used to."""
    n = man["tokens_written"] if limit is None else min(limit, man["tokens_written"])
    nb = n // seq_len
    return (nb - max(1, int(nb * val_fraction))) * seq_len, nb * seq_len


LIMITS = {"cosmopedia": 403451249, "ultrafineweb_en": 256832885, "tinystories": 73377468}


def erank(sv):
    p = sv / sv.sum().clamp_min(1e-12)
    return float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))


def main():
    a = parse_args()
    tk, ids = build_tokenizer("allenai/OLMo-1B-hf")
    model, n_layers, _ = load_any(a.ckpt, ids)
    step = torch.load(a.ckpt, map_location="cpu").get("step", 0)

    names = [s.strip() for s in a.corpora.split(",") if s.strip()]
    per = max(1, a.seqs // len(names))
    chunks = []
    for nm in names:
        man = json.load(open(f"data/{nm}.manifest.json"))
        mm = np.memmap(f"data/{nm}.bin", dtype=np.uint16, mode="r",
                       shape=(man["target_tokens"],))
        chunks.append(torch.from_numpy(
            np.asarray(mm[_lo(man, nm, per):_lo(man, nm, per) + per * 128])
            .astype(np.int64)).view(-1, 128))
    x_ids = torch.cat(chunks)
    B, T = x_ids.shape

    rows = []
    with torch.no_grad():
        x = model.encoder.pe(model.encoder.embedding(x_ids))
        for i, blk in enumerate(model.encoder.encoder_blocks):
            mha = blk.mha
            n_h, d_k = mha.n_heads, mha.d_k
            # recompute attention explicitly -- the fused SDPA path never materialises
            # the probability matrix, and the entropy is the point of this tool
            q = (x @ mha.W_q).view(B, T, n_h, d_k).transpose(1, 2)
            k = (x @ mha.W_k).view(B, T, n_h, d_k).transpose(1, 2)
            v = (x @ mha.W_v).view(B, T, n_h, mha.d_v).transpose(1, 2)
            att = torch.softmax(q @ k.transpose(-2, -1) / (d_k ** 0.5), dim=-1)   # B,h,T,T
            o = att @ v                                                            # B,h,T,d_v

            ent = -(att * att.clamp_min(1e-12).log()).sum(-1).mean((0, 2))         # per head
            am = att.argmax(-1).float()
            pos = torch.arange(T, dtype=torch.float).view(1, 1, T)
            offset = (am - pos).abs().mean((0, 2))

            contrib = []
            for h in range(n_h):
                blockW = mha.W_O[h * mha.d_v:(h + 1) * mha.d_v, :].float()
                contrib.append(float(o[:, h].reshape(-1, mha.d_v).float().std(0).mean()
                                     * blockW.norm()))
            contrib = torch.tensor(contrib)
            p90 = torch.quantile(contrib, 0.90).clamp_min(1e-12)
            dead = int((contrib < 0.01 * p90).sum())

            attn_out = mha(x, False)
            # A freshly grafted layer has W_O identically zero, so the block writes
            # nothing and out_rank / contrib are zero by CONSTRUCTION. Entropy and
            # head_sim stay meaningful -- they come from Q and K, which the graft
            # re-initialised rather than zeroed.
            zeroed = bool(mha.W_O.abs().max() <= 0)
            A = attn_out.reshape(-1, attn_out.shape[-1]).float()
            A = A - A.mean(0, keepdim=True)
            out_rank = float("nan") if zeroed else erank(torch.linalg.svdvals(A[:2048]))

            # Head similarity must be CENTRED across heads. Raw cosine between
            # attention rows has a null of ~0.97, not 0: an untrained model puts every
            # head at near-uniform attention (entropy 4.82 of a 4.85 max), and the
            # shared 1/T floor dominates the cosine, so random init reads sim 0.95-0.99.
            # Measured on a fresh BertV2: raw 0.95-0.99 per layer, centred 0.00.
            # Subtracting the across-head mean removes whatever all heads share and
            # restores a null of zero. Raw is kept only for continuity with old rows.
            flat = att.mean(0).reshape(n_h, -1)
            fr = F.normalize(flat, dim=-1)
            sim_raw = (fr @ fr.T).fill_diagonal_(0).max().item()
            fc = F.normalize(flat - flat.mean(0, keepdim=True), dim=-1)
            # -inf, not 0, on the diagonal: centred heads sum to zero, so with no
            # duplication every real pair is ANTI-correlated and a 0 diagonal would both
            # win the argmax (reporting the nonsense pair h0~h0) and clamp the reported
            # max up to 0.00, hiding how far from duplicated the layer actually is.
            S = (fc @ fc.T).fill_diagonal_(float("-inf"))
            sim = S.max().item()
            j = int(S.argmax())
            dup = (j // n_h, j % n_h)          # which pair is actually duplicated

            rows.append(dict(layer=i, ent=[float(e) for e in ent],
                             contrib=[float(c) for c in contrib],
                             offset=[float(f) for f in offset],
                             head_dead=dead, out_rank=out_rank, head_sim=sim,
                             head_sim_raw=sim_raw, dup_pair=list(dup),
                             zeroed=zeroed, ceiling=n_h * d_k))
            x = blk.layer_norm_2(blk.feed_forward(blk.layer_norm_1(attn_out + x))
                                 + blk.layer_norm_1(attn_out + x))

    with open(a.history, "a") as f:
        f.write(json.dumps(dict(ckpt=a.ckpt, step=step, layers=rows)) + "\n")

    maxent = float(np.log(T))
    if a.quiet:
        print(f"attn step {step} (max entropy {maxent:.2f}, out_rank ceiling "
              f"{rows[0]['ceiling']})")
        print("  " + " ".join(
            f"L{r['layer']} rank={'ZERO-OUT' if r['zeroed'] else format(r['out_rank'], '.0f')}"
            f" ent={'/'.join(f'{e:.1f}' for e in r['ent'])}"
            f" sim={r['head_sim']:.2f}(h{r['dup_pair'][0]}~h{r['dup_pair'][1]})" for r in rows))
        print("  sim is CENTRED across heads. Measured null at random init: -0.23..-0.07"
              " (raw would read ~0.97)")
        return

    print(f"[attn-health] {a.ckpt}  step {step}   max entropy = log(T) = {maxent:.2f}")
    print(f"{'layer':6} {'out_rank':>9} {'/ceil':>6} {'dead':>5} {'sim':>6} {'(raw)':>7} "
          f"{'dup':>7}   {'per-head entropy':22} {'per-head contrib':24} {'mean |offset|'}")
    for r in rows:
        e = "/".join(f"{v:.2f}" for v in r["ent"])
        c = "/".join(f"{v:.2f}" for v in r["contrib"])
        o = "/".join(f"{v:.0f}" for v in r["offset"])
        print(f"{r['layer']:<6} {r['out_rank']:9.1f} {r['out_rank']/r['ceiling']:6.2f} "
              f"{r['head_dead']:5d} {r['head_sim']:6.2f} {r['head_sim_raw']:7.2f} "
              f"{'h%d~h%d' % tuple(r['dup_pair']):>7}   {e:22} {c:24} {o}")
    print("\n  entropy near max => head is averaging, not selecting")
    print("  out_rank/ceil near 1 => the 256-wide attention is genuinely binding")
    print("  sim   CENTRED across heads. Null at random init = 0.00. Near 1 = a genuinely")
    print("        duplicated pair, named in 'dup'. Only the max pair is reported, so a")
    print("        layer can have three diverse heads and still read high.")
    print("  (raw) uncentred, for continuity only -- its null is ~0.97, since untrained")
    print("        heads are all near-uniform and therefore all mutually similar.")


if __name__ == "__main__":
    main()
