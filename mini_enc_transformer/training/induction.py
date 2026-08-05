"""Teach copy-from-context (induction) with a mild CPU fine-tune.

"I watched Batman. The name of the movie was ___" fails because the model has no
induction circuit: nothing in MLM pretraining rewards attending back to an earlier
span and copying it forward. We train that directly with synthetic templates where
the answer is *provably* present earlier in the context.

Templates carry TWO entities ({E} and {A}) referenced several times each, so the model
cannot succeed by "copy the one capitalised word" -- it has to work out *which* entity
this slot refers to. Single-entity templates were too easy and produced a model that
copied indiscriminately.

Mixed with a continuation objective as replay, drawn from BOTH TinyStories and
UltraFineWeb, matching the phase-3 pretraining mixture -- training purely on templates
collapses the model onto them.

Branches from checkpoints/ckpt_autoreg/; writes a new file there. WSD so the branch stays extendable.
"""
import argparse
import json
import re
import time

import numpy as np
import torch

from mini_enc_transformer.model.mlm import IGNORE_INDEX
from mini_enc_transformer.training.pretrain import build_tokenizer, build_model

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=600)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--lr", type=float, default=3e-5)
p.add_argument("--copy-frac", type=float, default=0.25,
               help="share of induction examples. 0.5 made the model over-copy "
                    "('I read Dracula and Dracula'), so the default is lower.")
p.add_argument("--eos-frac", type=float, default=0.15)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--warmup-frac", type=float, default=0.10)
p.add_argument("--decay-frac", type=float, default=0.30)
p.add_argument("--min-ctx", type=int, default=8)
p.add_argument("--corpora", default="tinystories,ultrafineweb_en")
p.add_argument("--ckpt", default="checkpoints/ckpt_autoreg/ckpt_autoreg.pt")
p.add_argument("--out", default="checkpoints/ckpt_autoreg/ckpt_induction.pt")
a = p.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(0); np.random.seed(0)


class C:
    d_model, d_k, d_v, n_heads, n_layers, d_embed = 768, 64, 64, 4, 4, 128


SEQ, NM = 128, 3
tk, ids = build_tokenizer("allenai/OLMo-1B-hf")
MASK, EOS = ids["mask_id"], tk.eos_token_id
model = build_model(C(), ids)
model.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])
model.train()

# ---- replay corpora: mix, matching what phase 3 actually pretrained on -------
SCANS, EOSPOS = [], []
for name in [s.strip() for s in a.corpora.split(",") if s.strip()]:
    man = json.load(open(f"data/{name}.manifest.json"))
    mm = np.memmap(f"data/{name}.bin", dtype=np.uint16, mode="r", shape=(man["target_tokens"],))
    s = np.asarray(mm[:min(man["tokens_written"], 30_000_000)])
    e = np.flatnonzero(s == EOS)
    SCANS.append(s)
    EOSPOS.append(e[(e > SEQ) & (e < len(s) - 1)])
    print(f"replay corpus {name}: {len(s):,} tokens, {len(EOSPOS[-1]):,} boundaries", flush=True)

# ---- entity pool: real single-token capitalised words ------------------------
ENTS = [t for t in range(EOS)
        if (s := tk.decode([t])).startswith(" ") and s[1:].isalpha()
        and s[1:2].isupper() and 3 <= len(s[1:]) <= 12]
rng = np.random.default_rng(0)
rng.shuffle(ENTS)
HELD, TRAIN_ENTS = ENTS[:200], ENTS[200:]
print(f"entity pool: {len(TRAIN_ENTS):,} train / {len(HELD)} held out", flush=True)

# Two entities, each referenced several times, interleaved so position alone is no cue.
TEMPLATES = [
    "We went to{E} last year with{A}. I loved chatting with{E} while me and{A} walked around the city. I hope to visit{E} again soon.",
    "{E} and{A} were friends.{E} liked to run and{A} liked to swim. One day{E} asked{A} to play together.",
    "I watched{E} with{A}. The name of the movie was{E} and my friend was{A}. We both loved{E} a lot.",
    "The book{E} was written by{A}. Later{A} wrote a sequel to{E}, and everyone praised{A} for it.",
    "She met{E} at the park and{A} at the store. Later she called{E}, then she texted{A} about{E}.",
    "My dog{E} played with my cat{A}.{E} barked loudly while{A} slept. Then{E} ran to find{A}.",
    "{A} gave{E} a present.{E} thanked{A} and told{A} that{E} was very happy.",
    "The team{E} beat the team{A}. Fans of{E} cheered while fans of{A} left early, but{E} celebrated.",
]

SPLIT = re.compile(r"(\{[EA]\})")


def slots(tmpl):
    """Placeholder occurrences that have an EARLIER occurrence of the SAME entity --
    only those are answerable by copying."""
    parts = SPLIT.split(tmpl)
    out, seen = [], set()
    for i, part in enumerate(parts):
        if part in ("{E}", "{A}"):
            if part in seen:
                out.append(i)
            seen.add(part)
    return parts, out


def copy_batch(bs):
    """One (template, slot) per batch so all rows share a length; single-token
    entities keep it exact."""
    tmpl = TEMPLATES[int(rng.integers(len(TEMPLATES)))]
    parts, cand = slots(tmpl)
    target_i = int(rng.choice(cand))
    xs, ys = [], []
    for _ in range(bs):
        e, a_ = (int(x) for x in rng.choice(TRAIN_ENTS, size=2, replace=False))
        sub = {"{E}": tk.decode([e]), "{A}": tk.decode([a_])}
        prefix = "".join(sub.get(p, p) for p in parts[:target_i])
        answer = e if parts[target_i] == "{E}" else a_
        ctx = tk(prefix, add_special_tokens=False)["input_ids"]
        xs.append(np.array(ctx + [MASK] * NM, dtype=np.int64))
        lab = np.full(len(ctx) + NM, IGNORE_INDEX, dtype=np.int64)
        lab[len(ctx)] = answer
        ys.append(lab)
    n = min(len(x) for x in xs)
    return (torch.from_numpy(np.stack([x[-n:] for x in xs])),
            torch.from_numpy(np.stack([y[-n:] for y in ys])))


def cont_batch(bs, ctx_len):
    """Continuation replay, sampled across all corpora."""
    xs, ys = [], []
    for _ in range(bs):
        k = int(rng.integers(len(SCANS)))
        scan, eos_at = SCANS[k], EOSPOS[k]
        if rng.random() < a.eos_frac and len(eos_at):
            tgt = int(rng.choice(eos_at))
        else:
            tgt = int(rng.integers(SEQ, len(scan) - 1))
        ctx = np.asarray(scan[tgt - ctx_len:tgt]).astype(np.int64)
        xs.append(np.concatenate([ctx, np.full(NM, MASK, dtype=np.int64)]))
        lab = np.full(ctx_len + NM, IGNORE_INDEX, dtype=np.int64)
        lab[ctx_len] = int(scan[tgt])
        ys.append(lab)
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def wsd(step, total, wf, df):
    w, d = int(wf * total), int(df * total)
    if step < w:
        return step / max(1, w)
    if step < total - d:
        return 1.0
    return max(0.0, (total - step) / max(1, d))


UNSEEN = [
    "The film{E} was great and{A} agreed. Everyone praised{E} but nobody praised{A}, so{E}",
    "{E} sat next to{A} on the bus.{A} smiled at{E} and then{A}",
    "Yesterday I met{E} and{A}. I called{E} first, and after that I called{A}, because{E}",
]


@torch.no_grad()
def copy_acc(templates, n=120):
    model.eval()
    ok = 0
    for i in range(n):
        tmpl = templates[i % len(templates)]
        parts, cand = slots(tmpl)
        target_i = cand[i % len(cand)]
        e, a_ = int(HELD[i % len(HELD)]), int(HELD[(i + 37) % len(HELD)])
        if e == a_:
            continue
        sub = {"{E}": tk.decode([e]), "{A}": tk.decode([a_])}
        prefix = "".join(sub.get(p, p) for p in parts[:target_i])
        answer = e if parts[target_i] == "{E}" else a_
        ctx = tk(prefix, add_special_tokens=False)["input_ids"]
        lg = model(torch.tensor([ctx + [MASK] * NM]))["logits"][0, len(ctx)]
        ok += int(int(lg.argmax()) == answer)
    model.train()
    return ok / n


print(f"BEFORE  trained-style {copy_acc(TEMPLATES):.3f} | unseen {copy_acc(UNSEEN):.3f}", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: wsd(s, a.steps, a.warmup_frac, a.decay_frac))

t0, run = time.time(), 0.0
for step in range(1, a.steps + 1):
    if rng.random() < a.copy_frac:
        x, y = copy_batch(a.batch)
    else:
        x, y = cont_batch(a.batch, int(rng.integers(a.min_ctx, SEQ - NM + 1)))
    loss = model(x, y)["loss"]
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    run += loss.item()
    if step % 100 == 0:
        print(f"step {step}/{a.steps} loss {run/100:.4f} lr {sched.get_last_lr()[0]:.2e} "
              f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
        run = 0.0

tr, un = copy_acc(TEMPLATES), copy_acc(UNSEEN)
model.eval()
torch.save({"model": model.state_dict(), "base": a.ckpt, "steps": a.steps,
            "copy_acc_trained": tr, "copy_acc_unseen": un, "copy_frac": a.copy_frac}, a.out)
print(f"AFTER   trained-style {tr:.3f} | unseen {un:.3f}", flush=True)
print(f"saved {a.out} in {(time.time()-t0)/60:.1f} min", flush=True)
