"""Is d_ff actually the constraint in the model being trained?

Measures the live checkpoint's own FFN: how many of its d_ff hidden directions carry
the activation variance, and how many units contribute negligibly. If r@99% sits near
d_ff the width is saturated and shrinking it cost real capacity; if it sits well below,
the width was never the binding constraint.

Forward passes only -- no training, no GPU.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, json
import numpy as np, torch, torch.nn.functional as F
from mini_enc_transformer.model.mlm import BertForMaskedLM as BertV1
from mini_enc_transformer.model_v2.mlm import BertForMaskedLM as BertV2
from mini_enc_transformer.training.pretrain import build_tokenizer


def load_any(path, ids):
    sd = torch.load(path, map_location="cpu")["model"]
    n_layers = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("encoder.encoder_blocks."))
    d_ff = sd["encoder.encoder_blocks.0.feed_forward.l1"].shape[1]
    cls = BertV1 if (n_layers == 4 and d_ff == 3072) else BertV2
    kw = dict(vocab_size=ids["vocab_size"], d_model=768, d_k=64, d_v=64, n_heads=4,
              n_layers=n_layers, pad_id=ids["pad_id"], d_embed=128)
    m = cls(**kw) if cls is BertV1 else cls(d_ff=d_ff, **kw)
    m.load_state_dict(sd); m.eval()
    return m, n_layers, d_ff, ("relu" if cls is BertV1 else "gelu"), sd


def eff_rank(sv):
    p = sv / sv.sum().clamp_min(1e-12)
    return float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))


def rank_for_energy(sv, target):
    e = (sv ** 2).cumsum(0) / (sv ** 2).sum()
    return int((e < target).sum().item()) + 1

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="checkpoints/ckpt_v2/last.pt")
p.add_argument("--corpora", default="cosmopedia,ultrafineweb_en,tinystories")
p.add_argument("--seqs", type=int, default=48)
a = p.parse_args()

tk, ids = build_tokenizer("allenai/OLMo-1B-hf")
m, n_layers, d_ff, act, _ = load_any(a.ckpt, ids)
step = torch.load(a.ckpt, map_location="cpu").get("step", "?")
print(f"[{a.ckpt}]  step {step}  |  {n_layers} layers, d_ff {d_ff}, {act}\n")

names = [s.strip() for s in a.corpora.split(",") if s.strip()]
per = a.seqs // len(names)
chunks = []
for nm in names:
    man = json.load(open(f"data/{nm}.manifest.json"))
    mm = np.memmap(f"data/{nm}.bin", dtype=np.uint16, mode="r", shape=(man["target_tokens"],))
    chunks.append(torch.from_numpy(np.asarray(mm[7_000_000:7_000_000 + per*128]).astype(np.int64)).view(-1,128))
ids_t = torch.cat(chunks)

print(f"{'layer':6} {'eff_rank':>9} {'r@90%':>7} {'r@99%':>7} {'r99/d_ff':>9} "
      f"{'|a|<1e-3':>9} {'top-10% share':>14}")
with torch.no_grad():
    x = m.encoder.pe(m.encoder.embedding(ids_t))
    for i, blk in enumerate(m.encoder.encoder_blocks):
        h = blk.layer_norm_1(blk.mha(x, False) + x)
        ff = blk.feed_forward
        pre = h @ ff.l1 + ff.b1
        act_fn = ff.act if hasattr(ff, "act") else ff.relu_1
        A = act_fn(pre).reshape(-1, d_ff).float()
        Ac = A - A.mean(0, keepdim=True)
        sv = torch.linalg.svdvals(Ac[:4096])
        contrib = A.abs().mean(0) * ff.l2.detach().float().norm(dim=1)
        top = torch.topk(contrib, max(1, d_ff // 10)).values.sum() / contrib.sum()
        print(f"{i:<6} {eff_rank(sv):9.1f} {rank_for_energy(sv,0.90):7d} "
              f"{rank_for_energy(sv,0.99):7d} {rank_for_energy(sv,0.99)/d_ff:9.2f} "
              f"{int((A.abs().max(0).values < 1e-3).sum()):9d} {100*top:13.1f}%")
        x = blk.layer_norm_2(ff(h) + h)
print("\nr99/d_ff near 1.00 => width saturated, the cut cost capacity.")
print("well below 1.00    => width still not the binding constraint.")
