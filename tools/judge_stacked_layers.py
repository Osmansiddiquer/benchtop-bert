"""Did the stacked layers ever become their own layers?

L4 was initialised as a copy of L2 and L5 as a copy of L3. Two ways that can fail:

  * they stay near-duplicates -- weight cosine to their parent remains high, while
    naturally-trained adjacent layers sit at cosine ~0.00 (measured on this model)
  * the model suppresses them -- pre-activations driven permanently negative, so the
    layer contributes nothing regardless of what its weights say

Exit code 0 = differentiated (keep the run), 1 = did not (worth re-initialising).

    python tools/judge_stacked_layers.py --ckpt checkpoints/ckpt_v2/last.pt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ckpt_v2/last.pt")
    p.add_argument("--pairs", default="4:2,5:3", help="child:parent")
    p.add_argument("--cos-max", type=float, default=0.50,
                   help="above this the child is still a copy of its parent")
    p.add_argument("--np-max", type=float, default=0.50,
                   help="never-positive fraction above this means the layer is suppressed")
    p.add_argument("--slack-history", default="logs/ffn_slack.jsonl")
    return p.parse_args()


TENSORS = ["mha.W_q", "mha.W_k", "mha.W_v", "mha.W_O",
           "feed_forward.l1", "feed_forward.l2"]


def main():
    a = parse_args()
    W = torch.load(a.ckpt, map_location="cpu")
    step = W.get("step", "?")
    W = W["model"]

    def blk(i):
        pre = f"encoder.encoder_blocks.{i}."
        return {k[len(pre):]: v for k, v in W.items() if k.startswith(pre)}

    # latest never-positive reading per layer, if the slack history exists
    npos = {}
    if os.path.exists(a.slack_history):
        rows = [json.loads(l) for l in open(a.slack_history) if l.strip()]
        rows = [r for r in rows if r.get("ckpt") == a.ckpt]
        if rows:
            npos = {x["layer"]: x["never_pos"] for x in rows[-1]["layers"]}

    print(f"[judge] {a.ckpt}  step {step}")
    print(f"{'pair':12} {'mean cos':>9} {'drift':>8} {'never+':>8}   verdict")
    verdicts = []
    for spec in a.pairs.split(","):
        c, p = (int(x) for x in spec.split(":"))
        cs, num, den = [], 0.0, 0.0
        for t in TENSORS:
            x, y = blk(p)[t].float(), blk(c)[t].float()
            cs.append(F.cosine_similarity(x.flatten().unsqueeze(0),
                                          y.flatten().unsqueeze(0)).item())
            num += (y - x).norm().item() ** 2
            den += x.norm().item() ** 2
        cos = sum(cs) / len(cs)
        drift = (num ** 0.5) / (den ** 0.5)
        np_c = npos.get(c, float("nan"))
        still_copy = cos > a.cos_max
        suppressed = (np_c == np_c) and np_c > a.np_max
        ok = not (still_copy or suppressed)
        why = "differentiated" if ok else \
              ("still a copy" if still_copy else "") + \
              (" + suppressed" if still_copy and suppressed else
               ("suppressed" if suppressed else ""))
        verdicts.append(ok)
        print(f"L{c} <- L{p}      {cos:9.4f} {drift:8.4f} {100*np_c:7.1f}%   {why}")

    ok = all(verdicts)
    print(f"\n  VERDICT: {'KEEP' if ok else 'REINIT'} "
          f"(thresholds: cos<{a.cos_max}, never+<{a.np_max})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
