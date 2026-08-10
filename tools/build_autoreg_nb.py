"""Regenerate notebooks/autoreg.ipynb from source held here.

The notebook is a deliverable but a bad place to keep code under revision: JSON diffs
are unreadable and cell edits are easy to lose. This script is the source of truth --
edit here, re-run, and the notebook is rebuilt deterministically.

    python tools/build_autoreg_nb.py
"""
import json
import os

MD = "markdown"
PY = "code"


# ----------------------------------------------------------------------------------
INTRO = r"""# Coercing an MLM encoder into an autoregressive generator

The models here were trained with **masked language modelling** — bidirectional, no causal
mask, no next-token objective. They are not generative models. This notebook turns one into
a passable generator and then teaches it to copy from context.

**Decoding.** Append `N_MASKS` trailing `[MASK]` tokens, read the prediction at the **first**
slot, commit it, repeat. The trailing masks matter: a mask at the very end of a sequence is a
distribution the model rarely saw (masks were sprinkled *inside* packed text), so a couple of
extra masks make the position look more like training data.

**Why it needs fine-tuning.** Pretraining taught the model to *infill* masks inside packed
text; we are asking it to *continue* text. Different conditionals, and nothing in pretraining
rewarded advancing a sequence — so greedy decoding collapses into loops. Two fixes, both in
`train_autoreg`:

1. **Randomised context length** — at inference the window grows (the first generated token
   sees only the prompt, later ones see the full 125). Training at one fixed length leaves the
   model badly calibrated for short contexts.
2. **EOS oversampling** — `<|endoftext|>` separates documents and is rarely a masked target,
   so the model never learned where text ends. A fraction of examples are drawn at real
   document boundaries.

**Induction.** `"I watched Batman. The name of the movie was ___"` fails at ~1% — a *copying*
failure, not a knowledge failure. Induction is normally a two-layer circuit (previous-token
head feeding a copy head), and with 4 heads per layer there is little room for one to form
incidentally. Phase 2 mixes in synthetic two-entity templates where the answer is recoverable
only by attending back to an earlier occurrence.

---

## Layout

| cell | what |
|---|---|
| **Setup** | tokenizer, constants, paths |
| **The training cell** | `train_autoreg(...)` and everything it needs — the only place logic lives |
| the rest | call it on a checkpoint, generate from the prompt set |
"""

SETUP = r'''import os, sys, re, time, json, numpy as np, torch

# Run from either the repo root or notebooks/; data/ and checkpoints/ paths are repo-relative.
ROOT = os.path.abspath("." if os.path.isdir("mini_enc_transformer") else "..")
sys.path.insert(0, ROOT); os.chdir(ROOT)

from mini_enc_transformer import build_tokenizer, IGNORE_INDEX
from mini_enc_transformer.model.mlm import BertForMaskedLM as BertV1      # 4L, d_ff 3072, ReLU
from mini_enc_transformer.model_v2.mlm import BertForMaskedLM as BertV2   # 6L, d_ff 1792, GELU

SEQ_LEN  = 128   # context length the models were trained on; the sliding window matches it
N_MASKS  = 3     # trailing [MASK] tokens; only the first is ever committed

# Set ONCE here and passed explicitly everywhere below. train_autoreg, generate,
# copy_acc and report all take `device` -- if one call gets a different value from the
# model's actual placement you get "Expected all tensors to be on the same device".
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tk, ids = build_tokenizer("allenai/OLMo-1B-hf")
MASK_ID, EOS_ID = ids["mask_id"], tk.eos_token_id
torch.set_num_threads(4)          # leave headroom in case a GPU run is in progress

print(f"vocab {ids['vocab_size']} | mask_id {MASK_ID} | eos/doc-sep {EOS_ID}")

PROMPTS = [
    "Once upon a time there was a little girl who",
    "The movie was",
    "The capital of France is",
    "I watched this film last night and",
    "Goku",
]

# Prompts that specifically require copying an entity out of context.
INDUCTION_PROMPTS = [
    "I watched Batman. The name of the movie was",
    "My friend Sarah came over. I said hello to",
    "The book was called Dracula. I read",
    "Yesterday, Osman went to meet his friend Imran. Imran was",
]
'''

TRAIN_CELL = r'''# ===================================================================================
#  THE TRAINING CELL — all logic lives here. Everything below just calls train_autoreg.
# ===================================================================================

# ---- model loading: architecture is detected, not assumed ------------------------
def load_model(path, device=DEVICE):
    """Infer the architecture from tensor shapes so v1 (4L/3072/ReLU) and v2
    (6L/1792/GELU) checkpoints both load without the caller knowing which is which."""
    sd = torch.load(path, map_location="cpu")["model"]
    n_layers = 1 + max(int(k.split(".")[2]) for k in sd
                       if k.startswith("encoder.encoder_blocks."))
    d_ff = sd["encoder.encoder_blocks.0.feed_forward.l1"].shape[1]
    cls = BertV1 if (n_layers == 4 and d_ff == 3072) else BertV2
    kw = dict(vocab_size=ids["vocab_size"], d_model=768, d_k=64, d_v=64, n_heads=4,
              n_layers=n_layers, pad_id=ids["pad_id"], d_embed=128)
    model = cls(**kw) if cls is BertV1 else cls(d_ff=d_ff, **kw)
    model.load_state_dict(sd)
    return model.to(device), dict(n_layers=n_layers, d_ff=d_ff,
                                  act="relu" if cls is BertV1 else "gelu")


# ---- replay corpora: weighted, shuffled, memmapped once, cached ------------------
def parse_corpora(spec):
    """Accept any of:
        ("tinystories", "ultrafineweb_en")            equal weight
        ("tinystories:0.15", "ultrafineweb_en:0.85")  explicit
        {"tinystories": 0.15, "ultrafineweb_en": 0.85}

    Weights are SAMPLING PROBABILITIES per example, not token counts. Uniform mixing
    means a corpus contributes equally regardless of size or quality -- and TinyStories
    at 50% visibly drags the register down, because it was built with a ~1,500-word
    vocabulary at a 3-4-year-old reading level. Down-weight it.
    """
    if isinstance(spec, dict):
        names = list(spec.keys())
        w = np.asarray([float(v) for v in spec.values()], dtype=float)
    else:
        names, ws = [], []
        for item in spec:
            if isinstance(item, (tuple, list)):
                n, wt = item[0], float(item[1])
            elif ":" in str(item):
                n, wt = str(item).rsplit(":", 1)
                wt = float(wt)
            else:
                n, wt = str(item), 1.0
            names.append(n); ws.append(wt)
        w = np.asarray(ws, dtype=float)
    if w.sum() <= 0:
        raise ValueError(f"corpus weights must sum > 0, got {w}")
    return names, w / w.sum()


_REPLAY = {}

def replay_corpora(names, n_tokens, shuffle=True, seed=0):
    """Load a window of each corpus into RAM.

    shuffle=True draws the window from a RANDOM offset instead of always the head, so
    repeated runs replay different text rather than the same first n_tokens every time.
    The offset is seeded, so a given seed is still reproducible.
    """
    key = (tuple(names), n_tokens, shuffle, seed)
    if key in _REPLAY:
        return _REPLAY[key]
    off_rng = np.random.default_rng(seed)
    scans, eospos = [], []
    for name in names:
        man = json.load(open(f"data/{name}.manifest.json"))
        mm = np.memmap(f"data/{name}.bin", dtype=np.uint16, mode="r",
                       shape=(man["target_tokens"],))
        avail = man["tokens_written"]
        take = min(avail, n_tokens)
        lo = int(off_rng.integers(0, avail - take + 1)) if shuffle else 0
        s = np.asarray(mm[lo:lo + take])
        e = np.flatnonzero(s == EOS_ID)
        scans.append(s)
        eospos.append(e[(e > SEQ_LEN) & (e < len(s) - 1)])
        print(f"  replay {name}: {len(s):,} tokens from offset {lo:,} "
              f"| {len(eospos[-1]):,} doc boundaries")
    _REPLAY[key] = (scans, eospos)
    return _REPLAY[key]


def make_lm_batch(rng, scans, eospos, weights, bs, ctx_len, eos_frac):
    """Left context + N_MASKS masks; supervise ONLY the first slot with the true next
    token. That is next-token prediction expressed through the mask interface.

    One context length per batch so rows stack without padding -- the model has no
    key-padding mask, so left-padding would inject tokens it would wrongly attend to.
    """
    xs, ys = [], []
    for _ in range(bs):
        k = int(rng.choice(len(scans), p=weights))   # WEIGHTED corpus choice per example
        scan, eos_at = scans[k], eospos[k]
        if rng.random() < eos_frac and len(eos_at):
            tgt = int(rng.choice(eos_at))            # next token IS eos -> teaches stopping
        else:
            tgt = int(rng.integers(SEQ_LEN, len(scan) - 1))
        ctx = np.asarray(scan[tgt - ctx_len:tgt]).astype(np.int64)
        xs.append(np.concatenate([ctx, np.full(N_MASKS, MASK_ID, dtype=np.int64)]))
        lab = np.full(ctx_len + N_MASKS, IGNORE_INDEX, dtype=np.int64)
        lab[ctx_len] = int(scan[tgt])
        ys.append(lab)
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


# ---- induction: two-entity templates --------------------------------------------
# TWO entities, interleaved. With a single entity the task collapses to "copy the one
# capitalised word" -- the model solves that instantly and then over-copies everywhere.
# Two force it to resolve WHICH one a slot refers to, which is the actual skill.
TEMPLATES = [
    'We went to{E} last year with{A}. I loved chatting with{E} while me and{A} walked around the city. I hope to visit{E} again soon.',
    '{E} and{A} were friends.{E} liked to run and{A} liked to swim. One day{E} asked{A} to play together.',
    'I watched{E} with{A}. The name of the movie was{E} and my friend was{A}. We both loved{E} a lot.',
    'The book{E} was written by{A}. Later{A} wrote a sequel to{E}, and everyone praised{A} for it.',
    'She met{E} at the park and{A} at the store. Later she called{E}, then she texted{A} about{E}.',
    'My dog{E} played with my cat{A}.{E} barked loudly while{A} slept. Then{E} ran to find{A}.',
    '{A} gave{E} a present.{E} thanked{A} and told{A} that{E} was very happy.',
    'The team{E} beat the team{A}. Fans of{E} cheered while fans of{A} left early, but{E} celebrated.',
]

# Phrasings never trained on -- the honest test of whether a copy circuit formed rather
# than the templates being memorised.
UNSEEN_TEMPLATES = [
    'The film{E} was great and{A} agreed. Everyone praised{E} but nobody praised{A}, so{E}',
    '{E} sat next to{A} on the bus.{A} smiled at{E} and then{A}',
    'Yesterday I met{E} and{A}. I called{E} first, and after that I called{A}, because{E}',
]

_SPLIT = re.compile(r"(\{[EA]\})")

def slots(tmpl):
    """Placeholder occurrences that have an EARLIER occurrence of the SAME entity --
    only those are answerable by copying."""
    parts = _SPLIT.split(tmpl)
    out, seen = [], set()
    for i, part in enumerate(parts):
        if part in ("{E}", "{A}"):
            if part in seen:
                out.append(i)
            seen.add(part)
    return parts, out


_ENTS = None

def entity_pools(seed=0, n_held=200):
    """Single-token capitalised words, split train / held-out so evaluation measures
    COPYING rather than memorising particular names. Computed once."""
    global _ENTS
    if _ENTS is None:
        pool = [t for t in range(EOS_ID)
                if (s := tk.decode([t])).startswith(" ") and s[1:].isalpha()
                and s[1:2].isupper() and 3 <= len(s[1:]) <= 12]
        np.random.default_rng(seed).shuffle(pool)
        _ENTS = (pool[n_held:], pool[:n_held])       # (train, held-out)
        print(f"  entity pool: {len(_ENTS[0]):,} train / {len(_ENTS[1])} held out")
    return _ENTS


def make_copy_batch(rng, train_ents, bs):
    """One (template, slot) per batch so rows share a length; single-token entities keep
    it exact. The answer is whichever entity that slot refers to -- recoverable only by
    attending back to its earlier occurrence."""
    tmpl = TEMPLATES[int(rng.integers(len(TEMPLATES)))]
    parts, cand = slots(tmpl)
    target_i = int(rng.choice(cand))
    xs, ys = [], []
    for _ in range(bs):
        e, a_ = (int(x) for x in rng.choice(train_ents, size=2, replace=False))
        sub = {"{E}": tk.decode([e]), "{A}": tk.decode([a_])}
        prefix = "".join(sub.get(p, p) for p in parts[:target_i])
        answer = e if parts[target_i] == "{E}" else a_
        ctx = tk(prefix, add_special_tokens=False)["input_ids"]
        xs.append(np.array(ctx + [MASK_ID] * N_MASKS, dtype=np.int64))
        lab = np.full(len(ctx) + N_MASKS, IGNORE_INDEX, dtype=np.int64)
        lab[len(ctx)] = answer
        ys.append(lab)
    n = min(len(x) for x in xs)
    return (torch.from_numpy(np.stack([x[-n:] for x in xs])),
            torch.from_numpy(np.stack([y[-n:] for y in ys])))


def wsd(step, total, warmup_frac=0.10, decay_frac=0.30):
    """Warmup-Stable-Decay (MiniCPM 2024): ramp, hold, decay only at the end."""
    w, d = int(warmup_frac * total), int(decay_frac * total)
    if step < w:
        return step / max(1, w)
    if step < total - d:
        return 1.0
    return max(0.0, (total - step) / max(1, d))


# ---- the entry point -------------------------------------------------------------
def train_autoreg(ckpt_path, out_path, *,
                  steps_lm=400, steps_ind=600, batch=8, lr=3e-5,
                  eos_frac=0.30, copy_frac=0.25, min_ctx=8,
                  corpora=("ultrafineweb_en:0.85", "tinystories:0.15"),
                  replay_tokens=30_000_000, shuffle_replay=True, seed=0, device=DEVICE,
                  log_every=100, verbose=True):
    """Turn an MLM encoder into an autoregressive generator with a copy circuit.

    Two phases, each with its own WSD schedule:

      1. steps_lm   next-token-through-mask only. Randomised context length, EOS
                    oversampled at `eos_frac`. Fixes the looping.
      2. steps_ind  the same, with `copy_frac` of batches replaced by two-entity
                    induction templates. Builds the copy circuit.

    Phase 2 keeps replaying phase-1 data: training on copy alone makes the model
    over-copy ("I read Dracula and Dracula"). copy_frac=0.5 was measured to do this;
    0.25 is the working value.

    `corpora` takes weights ("name:weight", or a dict). They are per-example sampling
    probabilities. The default down-weights TinyStories to 15%: it was built with a
    ~1,500-word vocabulary at a 3-4-year-old reading level, so replaying it at parity
    pulls the generator's whole register down to children's-story English. Raise it only
    if you want that. For a v2 checkpoint, "cosmopedia" is also in-distribution and is
    much better prose than TinyStories.

    Returns the trained model in eval mode, and writes it to out_path.
    Set steps_ind=0 to skip induction, steps_lm=0 to do induction only.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    model, arch = load_model(ckpt_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[autoreg] {ckpt_path}")
    print(f"  arch {arch['n_layers']}L d_ff={arch['d_ff']} {arch['act'].upper()} "
          f"| {n_params/1e6:.1f}M params | device {device}")

    names, weights = parse_corpora(corpora)
    print("  replay mix: " + ", ".join(f"{n} {w:.0%}" for n, w in zip(names, weights)))
    scans, eospos = replay_corpora(names, replay_tokens, shuffle_replay, seed)
    train_ents, _ = entity_pools(seed)

    def run_phase(name, steps, use_copy):
        if steps <= 0:
            print(f"  [{name}] skipped")
            return
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: wsd(s, steps))
        t0, run = time.time(), 0.0
        for step in range(1, steps + 1):
            if use_copy and rng.random() < copy_frac:
                x, y = make_copy_batch(rng, train_ents, batch)
            else:
                L = int(rng.integers(min_ctx, SEQ_LEN - N_MASKS + 1))
                x, y = make_lm_batch(rng, scans, eospos, weights, batch, L, eos_frac)
            loss = model(x.to(device), y.to(device))["loss"]
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run += loss.item()
            if verbose and step % log_every == 0:
                print(f"  [{name}] step {step}/{steps} loss {run/log_every:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} "
                      f"({(time.time()-t0)/step:.2f}s/step)")
                run = 0.0
        print(f"  [{name}] done in {(time.time()-t0)/60:.1f} min")

    run_phase("lm ", steps_lm, use_copy=False)
    run_phase("ind", steps_ind, use_copy=True)

    model.eval()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save({"model": model.state_dict(), "base": ckpt_path, "arch": arch,
                "recipe": dict(steps_lm=steps_lm, steps_ind=steps_ind, batch=batch,
                               lr=lr, eos_frac=eos_frac, copy_frac=copy_frac,
                               min_ctx=min_ctx, corpora=list(corpora),
                               weights=weights.tolist(), shuffle_replay=shuffle_replay,
                               seed=seed)},
               out_path)
    print(f"  saved -> {out_path}")
    return model


# ---- inference -------------------------------------------------------------------
@torch.no_grad()
def generate(model, prompt, max_gen=30, temperature=0.0, top_k=None,
             stop_on={".", "!", "?"}, device=DEVICE):
    """Mask-fill decoding: append N_MASKS masks, commit the FIRST slot, repeat.

    add_special_tokens=False is essential -- otherwise the tokenizer appends
    <|endoftext|>, the model reads a document boundary, and generation starts an
    unrelated new document instead of continuing the sentence.
    """
    was_training = model.training
    model.eval()
    toks = tk(prompt, add_special_tokens=False)["input_ids"]
    for _ in range(max_gen):
        ctx = toks[-(SEQ_LEN - N_MASKS):]
        x = torch.tensor([ctx + [MASK_ID] * N_MASKS], device=device)
        logits = model(x)["logits"][0, len(ctx)]
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            z = logits / temperature
            if top_k:
                z[z < torch.topk(z, top_k).values[-1]] = -float("inf")
            nxt = int(torch.multinomial(torch.softmax(z, -1), 1))
        if nxt == EOS_ID:
            break
        toks.append(nxt)
        if tk.decode([nxt]).strip() in stop_on:
            break
    if was_training:
        model.train()
    return tk.decode(toks)


@torch.no_grad()
def copy_acc(model, templates, n=120, seed=0, device=DEVICE):
    """Copy accuracy on entities the model never trained on."""
    _, held = entity_pools(seed)
    was_training = model.training
    model.eval()
    ok = 0
    for i in range(n):
        tmpl = templates[i % len(templates)]
        parts, cand = slots(tmpl)
        target_i = cand[i % len(cand)]
        e, a_ = int(held[i % len(held)]), int(held[(i + 37) % len(held)])
        if e == a_:
            continue
        sub = {"{E}": tk.decode([e]), "{A}": tk.decode([a_])}
        prefix = "".join(sub.get(p, p) for p in parts[:target_i])
        answer = e if parts[target_i] == "{E}" else a_
        ctx = tk(prefix, add_special_tokens=False)["input_ids"]
        lg = model(torch.tensor([ctx + [MASK_ID] * N_MASKS], device=device))["logits"][0, len(ctx)]
        ok += int(int(lg.argmax()) == answer)
    if was_training:
        model.train()
    return ok / n


def report(model, name, device=DEVICE):
    """Prompt sweep + copy accuracy for one trained model."""
    print(f"\n{'='*70}\n{name}  [device {device}]\n{'='*70}")
    for p in PROMPTS:
        print(f"> {p}\n  {generate(model, p, device=device)}\n")
    print("-- induction prompts --")
    for p in INDUCTION_PROMPTS:
        print(f"> {p}\n  {generate(model, p, max_gen=20, device=device)}\n")
    print(f"copy acc, TRAINED templates / held-out entities : "
          f"{copy_acc(model, TEMPLATES, device=device):.3f}")
    print(f"copy acc, UNSEEN  templates / held-out entities : "
          f"{copy_acc(model, UNSEEN_TEMPLATES, device=device):.3f}")
'''

USAGE_MD = r"""## Using it

```python
model = train_autoreg("checkpoints/ckpt3/last.pt",
                      "checkpoints/ckpt_autoreg/ckpt3_autoreg.pt")
report(model, "ckpt3")
```

Knobs worth touching:

| knob | default | why you'd change it |
|---|---|---|
| `corpora` | `("ultrafineweb_en:0.85","tinystories:0.15")` | **weights are per-example sampling probabilities.** TinyStories has a ~1,500-word vocabulary — at parity it drags the register to children's-story English |
| `shuffle_replay` | `True` | draw the replay window from a random (seeded) offset instead of always the head |
| `steps_lm` / `steps_ind` | 400 / 600 | more of phase 2 if copy accuracy is still low |
| `copy_frac` | 0.25 | **0.5 measured to over-copy** ("I read Dracula and Dracula") |
| `eos_frac` | 0.30 | lower if generations stop too early |
| `min_ctx` | 8 | the short end of the randomised context range |
| `device` | `"cpu"` | `"cuda"` if the GPU is free — ~10× faster |
| `steps_ind=0` | — | phase 1 only, to isolate the looping fix |

Each call below is independent: a base checkpoint in, a trained model out.
"""

BASELINE = r'''# Baseline: what the raw MLM encoder does before any of this. Expect loops.
base, arch = load_model("checkpoints/ckpt3/last.pt")
print(f"ckpt3: {arch['n_layers']}L d_ff={arch['d_ff']} {arch['act']}")
for p in PROMPTS[:3]:
    print(f"> {p}\n  {generate(base, p)}\n")
print(f"copy acc (trained templates): {copy_acc(base, TEMPLATES):.3f}   <- ~0.01, cannot copy")
del base
'''

CALL_CKPT3 = r'''# --- phase-3 baseline encoder: 4 layers, d_ff 3072, ReLU ---------------------------
m_ckpt3 = train_autoreg(
    "checkpoints/ckpt3/last.pt",
    "checkpoints/ckpt_autoreg/ckpt3_autoreg.pt",
)
report(m_ckpt3, "ckpt3 -> autoreg + induction")
'''

CALL_V2 = r'''# --- v2 architecture: 6 layers, d_ff 1792, GELU -----------------------------------
m_v2 = train_autoreg(
    "checkpoints/ckpt_v2/pre_anneal.pt",
    "checkpoints/ckpt_autoreg/v2_pre_anneal_autoreg.pt",
)
report(m_v2, "v2 pre-anneal -> autoreg + induction")
'''

CALL_V2_ANNEAL = r'''# --- v2 after the cosmopedia anneal ------------------------------------------------
m_v2a = train_autoreg(
    "checkpoints/ckpt_v2_anneal_cosmo/last.pt",
    "checkpoints/ckpt_autoreg/v2_cosmo_autoreg.pt",
)
report(m_v2a, "v2 cosmopedia-annealed -> autoreg + induction")
'''

ABLATION = r'''# --- ablation: phase 1 only, no induction ------------------------------------------
# Isolates the looping fix from the copy circuit. Copy accuracy should stay near zero
# while the generations stop looping.
m_lm_only = train_autoreg(
    "checkpoints/ckpt_v2/pre_anneal.pt",
    "checkpoints/ckpt_autoreg/v2_lm_only.pt",
    steps_ind=0,
)
report(m_lm_only, "v2 pre-anneal -> phase 1 only (no induction)")
'''

COMPARE = r'''# --- side-by-side copy accuracy ----------------------------------------------------
# UNSEEN templates are the honest number: TRAINED-template accuracy can be inflated by
# memorising the phrasing rather than learning to copy.
for name, m in [("ckpt3", m_ckpt3), ("v2 pre-anneal", m_v2),
                ("v2 cosmo-annealed", m_v2a), ("v2 lm-only", m_lm_only)]:
    print(f"{name:<22} trained {copy_acc(m, TEMPLATES):.3f}   "
          f"unseen {copy_acc(m, UNSEEN_TEMPLATES):.3f}")
'''

SAMPLING = r'''# --- sampling, for reference -------------------------------------------------------
# Greedy loops more than sampling does; sampling trades coherence for variety. The gap is
# exaggerated here because there is no next-token objective holding the sequence together.
torch.manual_seed(0)
for p in PROMPTS:
    print(f"> {p}")
    print(f"  greedy : {generate(m_v2a, p)}")
    for t in (0.7, 1.0):
        print(f"  T={t}    : {generate(m_v2a, p, temperature=t, top_k=40)}")
    print()
'''

NOTES = r"""## Reading the output

- **Fluent-but-looping is the failure mode phase 1 fixes.** If loops survive it, raise
  `steps_lm` or check that context length is actually being randomised.
- **UNSEEN-template copy accuracy is the number that matters.** High accuracy on TRAINED
  templates with low accuracy on unseen ones means the phrasings were memorised and no copy
  circuit formed.
- **Story prompts do best** — the replay mix is TinyStories + UltraFineWeb, so
  "Once upon a time..." continues far more sensibly than a factual prompt. That is the
  training distribution showing through, not reasoning.
- **This is not a capability benchmark.** It is a qualitative probe of what the encoder
  absorbed. The quantitative measures are held-out MLM accuracy and the SST-2 fine-tune.

Every model above is saved under `checkpoints/ckpt_autoreg/`, with the base checkpoint and
the full recipe recorded in the file so a run can be traced back.
"""


CELLS = [
    (MD, INTRO),
    (PY, SETUP),
    (PY, TRAIN_CELL),
    (MD, USAGE_MD),
    (PY, BASELINE),
    (PY, CALL_CKPT3),
    (PY, CALL_V2),
    (PY, CALL_V2_ANNEAL),
    (PY, ABLATION),
    (PY, COMPARE),
    (PY, SAMPLING),
    (MD, NOTES),
]


def cell(kind, src):
    src = src.rstrip("\n")
    lines = [l + "\n" for l in src.split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    base = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == PY:
        base["execution_count"] = None
        base["outputs"] = []
    return base


MARKER = "THE TRAINING CELL"


def sync(path):
    """Replace ONLY the training cell, in place, leaving every other cell and all
    outputs untouched.

    Jupyter writes the whole notebook on save, so a full rebuild silently destroys
    hand-edits to the call cells (step counts, device, extra cells). This finds the
    logic cell by its marker comment and swaps just that source.
    """
    with open(path) as f:
        nb = json.load(f)
    hits = [i for i, c in enumerate(nb["cells"])
            if c["cell_type"] == PY and MARKER in "".join(c["source"])]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly 1 cell containing {MARKER!r}, found {len(hits)}")
    i = hits[0]
    old = "".join(nb["cells"][i]["source"])
    nb["cells"][i] = cell(PY, TRAIN_CELL)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")
    print(f"synced training cell {i} in {path}: {len(old)} -> {len(TRAIN_CELL)} chars")
    print(f"  {len(nb['cells'])} cells total, all others untouched")
    print("  NOTE: reload the notebook in Jupyter before running, or its in-memory")
    print("        copy will overwrite this on the next save.")


def main():
    import sys as _sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "notebooks", "autoreg.ipynb")
    if "--sync" in _sys.argv:
        return sync(out)
    nb = {
        "cells": [cell(k, s) for k, s in CELLS],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")
    print(f"wrote {out}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
