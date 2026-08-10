"""Is d_ff running out? Per-layer slack, utilisation and trend, appended to JSONL.

One forward pass over a FIXED held-out batch, capturing the tensor fed to each
down-projection. That tensor is h regardless of activation or gating, so this works
unchanged if the FFN ever becomes GLU-style.

Per neuron i in layer l:

    c_i = std_t(h_i) * ||W_out[i, :]||

std, not mean: a neuron with a large constant output is a bias, not a feature. Times the
output column norm: small activations written through a big vector still move the
residual stream. Counting zeros is meaningless under GELU -- its floor is ~-0.17 and
nothing is ever exactly zero.

    slack        fraction of neurons with c_i < 0.01 * the layer's OWN median c
                 (relative, because scales drift across depth and over training)
    never_pos    fraction never driven above 0 -- separates optimisation failure
                 from genuine allocation slack
    utilisation  erank(H_alive) / n_alive, centred, alive neurons only, separators
                 dropped. Approaching 1 means no redundancy left to reclaim.
    kurtosis     of alive activations. Falling kurtosis is the earliest warning:
                 neurons being pressed from sparse detectors into dense superposed
                 duty, which hurts rare tokens before it shows in perplexity.

The absolute slack number has no trustworthy threshold. What matters is
  (1) CONTRAST between layers -- where the width actually wants to be, and
  (2) TREND -- utilisation only rises, so 30% slack now can be 2% by the end.

All correlational. A flag should be confirmed by zero-init width expansion: append
random rows to W_in and ZERO columns to W_out (output unchanged, loss continuous),
branch 1-2k steps against a control, and see whether the new columns grow.

    python tools/ffn_slack.py --ckpt checkpoints/ckpt_v2/last.pt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time

import numpy as np
import torch

from mini_enc_transformer.model.mlm import BertForMaskedLM as BertV1
from mini_enc_transformer.model_v2.mlm import BertForMaskedLM as BertV2
from mini_enc_transformer.training.pretrain import build_tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ckpt_v2/last.pt")
    p.add_argument("--corpora", default="cosmopedia,ultrafineweb_en,tinystories")
    p.add_argument("--seqs", type=int, default=32, help="FIXED batch, same every call")
    p.add_argument("--svd-rows", type=int, default=1024, help="rows sampled for erank")
    p.add_argument("--slack-rel", type=float, default=0.01, help="c_i < rel * layer median")
    p.add_argument("--history", default="logs/ffn_slack.jsonl")
    p.add_argument("--quiet", action="store_true", help="one summary line only")
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


def fixed_batch(names, seqs, eos_id):
    """Deterministic slice, identical on every call, so trends are comparable."""
    per = max(1, seqs // len(names))
    chunks = []
    for nm in names:
        man = json.load(open(f"data/{nm}.manifest.json"))
        mm = np.memmap(f"data/{nm}.bin", dtype=np.uint16, mode="r",
                       shape=(man["target_tokens"],))
        lo, hi = val_offset(man, LIMITS.get(nm))
        need = per * 128
        off = lo if hi - lo >= need else max(0, hi - need)
        chunks.append(torch.from_numpy(
            np.asarray(mm[off:off + per * 128]).astype(np.int64)).view(-1, 128))
    return torch.cat(chunks)


def verdict(slack, util, trend_steps_to_zero):
    """Slack alone is not enough. Low slack with LOW utilisation means every neuron
    contributes a little but they duplicate each other -- redundancy, not exhaustion.
    Width is only genuinely tight when few neurons are idle AND the survivors are
    doing independent work."""
    if slack < 0.02 and util > 0.80:
        return "STARVED"
    if slack < 0.05 and util > 0.60:
        return "TIGHT"
    if trend_steps_to_zero is not None and trend_steps_to_zero > 0:
        return "TIGHTENING"
    if util < 0.40:
        return "REDUNDANT"          # widely shared directions; width is not binding
    if slack > 0.30:
        return "SLACK"
    return "OK"


def main():
    a = parse_args()
    tk, ids = build_tokenizer("allenai/OLMo-1B-hf")
    model, n_layers, d_ff = load_any(a.ckpt, ids)
    step = torch.load(a.ckpt, map_location="cpu").get("step", 0)

    names = [s.strip() for s in a.corpora.split(",") if s.strip()]
    x_ids = fixed_batch(names, a.seqs, tk.eos_token_id)
    keep = (x_ids != tk.eos_token_id)          # drop document separators
    keep[:, 0] = False                         # and the first position

    rows = []
    with torch.no_grad():
        x = model.encoder.pe(model.encoder.embedding(x_ids))
        for i, blk in enumerate(model.encoder.encoder_blocks):
            h_in = blk.layer_norm_1(blk.mha(x, False) + x)
            ff = blk.feed_forward
            act_fn = ff.act if hasattr(ff, "act") else ff.relu_1
            H = act_fn(h_in @ ff.l1 + ff.b1)                       # input to W_out
            Hf = H[keep].float()                                   # (tokens, d_ff)

            out_norm = ff.l2.detach().float().norm(dim=1)          # ||W_out[i, :]||
            # A freshly grafted layer has W_out identically zero on purpose, so every
            # c_i is zero by CONSTRUCTION rather than by training. Slack, utilisation and
            # kurtosis are then undefined -- and worse, they read as slack 100% /
            # util 1.00 / kurt nan, which look like findings and poison the run-level
            # means. Report the layer as ZERO-OUT and leave it out of the aggregates
            # until its output projection lifts off zero. never_pos stays meaningful:
            # it describes pre-activations, which this does not touch.
            zeroed = bool(out_norm.max() <= 0)
            c = Hf.std(0) * out_norm
            med = c.median().clamp_min(1e-12)
            p90 = torch.quantile(c, 0.90).clamp_min(1e-12)
            # Median-anchored slack breaks on bimodal layers: when most neurons are
            # near-dead the median sits inside that mass, so almost nothing falls below
            # 1% of it and slack reads 0% on a mostly-idle layer (measured: layer 0,
            # p90/p50 = 55, slack 0% by median vs 13.8% by p90). p90 anchors to the
            # working population instead. skew flags when the two disagree.
            slack = float((c < a.slack_rel * p90).float().mean())
            slack_med = float((c < a.slack_rel * med).float().mean())
            skew = float(p90 / med)
            never_pos = float((Hf.max(0).values <= 0).float().mean())

            alive = c >= a.slack_rel * p90
            n_alive = int(alive.sum())
            Ha = Hf[:, alive]
            sub = Ha[torch.randperm(Ha.shape[0])[:a.svd_rows]]
            sub = sub - sub.mean(0, keepdim=True)
            util = erank(torch.linalg.svdvals(sub)) / max(1, n_alive)

            z = (Ha - Ha.mean(0, keepdim=True)) / Ha.std(0, keepdim=True).clamp_min(1e-6)
            kurt = float((z ** 4).mean())

            if zeroed:
                slack = slack_med = skew = util = kurt = float("nan")
                n_alive = 0
            rows.append(dict(layer=i, slack=slack, slack_med=slack_med, skew=skew,
                             never_pos=never_pos, util=util, kurtosis=kurt,
                             n_alive=n_alive, zeroed=zeroed))
            x = blk.layer_norm_2(ff(h_in) + h_in)

    # ---- trend from history ---------------------------------------------------
    hist = []
    if os.path.exists(a.history):
        for line in open(a.history):
            try:
                hist.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    live = [r for r in rows if not r["zeroed"]]          # aggregates skip zeroed layers
    hist = [r for r in hist if r.get("ckpt") == a.ckpt][-8:]
    trend, to_zero = None, None
    if len(hist) >= 3:
        xs = [r["step"] for r in hist] + [step]
        ys = [r["mean_slack"] for r in hist] + [float(np.mean([r["slack"] for r in live]))]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        den = sum((v - mx) ** 2 for v in xs)
        if den > 0:
            trend = sum((xs[k] - mx) * (ys[k] - my) for k in range(n)) / den   # slack per step
            # Only project exhaustion off a trend that is actually distinguishable from
            # zero. A -0.01pp/1k slope is noise, and dividing by it produced confident
            # nonsense ("exhausted ~13,623 steps") from a flat series.
            if trend * 1000 < -0.05:
                to_zero = int(ys[-1] / -trend)

    mean_slack = float(np.mean([r["slack"] for r in live])) if live else float("nan")
    mean_util = float(np.mean([r["util"] for r in live])) if live else float("nan")
    v = verdict(mean_slack, mean_util,
                to_zero if (to_zero is not None and to_zero < 22389 - step) else None)

    with open(a.history, "a") as f:
        f.write(json.dumps(dict(t=time.time(), ckpt=a.ckpt, step=step, d_ff=d_ff,
                                mean_slack=mean_slack, mean_util=mean_util,
                                verdict=v, layers=rows)) + "\n")

    if a.quiet:
        flag = lambda r: ("!" if r["skew"] > 10 else ("*" if r["slack"] < 0.02 and r["never_pos"] < 0.05 else " "))
        print(f"ffn step {step} d_ff={d_ff} [{v}] util {mean_util:.2f}"
              + (f" | slack {trend*1000:+.2f}pp/1k"
                 + (f", exhausted ~{to_zero:,} steps" if to_zero else "") if trend is not None else ""))
        print("  " + " ".join(
            (f"L{r['layer']}[ZERO-OUT] np={100*r['never_pos']:.0f}%" if r["zeroed"] else
             f"L{r['layer']}{flag(r)}slack={100*r['slack']:.0f}% np={100*r['never_pos']:.0f}% "
             f"u={r['util']:.2f} k={r['kurtosis']:.0f} skew={r['skew']:.0f}") for r in rows))
        print("  ! = bimodal dead-zone mass (not width pressure)   "
              "* = genuinely dense, width-pressure candidate")
        return

    print(f"[ffn-slack] {a.ckpt}  step {step}  d_ff {d_ff}\n")
    print(f"{'layer':6} {'slack':>7} {'p90/p50':>8} {'never+':>7} {'alive':>6} "
          f"{'utilisation':>12} {'kurtosis':>9}  note")
    for r in rows:
        note = ""
        if r["zeroed"]:
            note = "ZERO-OUT: W_out is identically zero (fresh graft); slack/util/kurt " \
                   "undefined and excluded from the means"
        elif r["skew"] > 10:
            note = "bimodal - dead-zone mass, not width pressure"
        elif r["slack"] < 0.02 and r["never_pos"] < 0.05:
            note = "<- genuinely dense: width-pressure candidate"
        print(f"{r['layer']:<6} {100*r['slack']:6.1f}% {r['skew']:8.1f} "
              f"{100*r['never_pos']:6.1f}% {r['n_alive']:6d} {r['util']:12.3f} "
              f"{r['kurtosis']:9.2f}  {note}")
    if live:
        lo = min(live, key=lambda r: r["slack"]); hi = max(live, key=lambda r: r["slack"])
        print(f"\n  contrast: least slack L{lo['layer']} ({100*lo['slack']:.1f}%) vs "
              f"most L{hi['layer']} ({100*hi['slack']:.1f}%) -> "
              f"{100*(hi['slack']-lo['slack']):.1f}pp spread")
    if trend is not None:
        print(f"  trend: {trend*1000:+.2f}pp slack per 1k steps"
              + (f", exhausted in ~{to_zero:,} steps" if to_zero else ""))
    else:
        print("  trend: needs >=3 history points")
    print(f"  VERDICT: {v}")


if __name__ == "__main__":
    main()
