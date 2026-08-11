# Grafts, Depth, and Diagnostics

**A 28M-parameter encoder under a hard parameter cap: what two weeks of laptop-scale
experiments established.**

*Osman Siddique and Claude · August 2026 · `benchtop-bert`*

> 📄 **The formal version is [`paper/report.pdf`](paper/report.pdf)** (LaTeX source
> in [`paper/report.tex`](paper/report.tex)). This file is the same content in Markdown for
> reading on GitHub. Section numbers match.

---

## Abstract

We built a BERT-style encoder from scratch — attention, blocks, embeddings and MLM head, with
only the tokenizer borrowed — and pretrained it on a single 4 GB laptop GPU under a hard
30M-parameter cap. The goal was to buy downstream accuracy with architecture and data
interventions rather than parameters.

**Several components worked, and are reported with controls** (§3): a staged data curriculum,
replay mixing (with a corpus held at 0% replay as the negative control, losing 0.95 nats
against 0.25 for replayed corpora), span masking, and a two-phase procedure that installs an
induction circuit into a masked language model, raising copy-from-context accuracy from 0.8%
to 93%.

**The headline goal nonetheless failed.** A 0.435-nat (16%) improvement in held-out MLM loss
produced no measurable SST-2 gain, because at 872 dev examples the measurement instrument
(SE ≈ 1.2pp) is coarser than every effect we produced.

**The principal technical contribution** is a measured correction to the folklore on
function-preserving layer grafting (§4). Zero-initialised grafting is usually said to be exact
only in pre-LN architectures. We show it is exact in **post-LN** as well, under two conditions
that are not usually stated: the block must be appended at the **top** of the stack, and its
LayerNorm affine parameters must be **reset** to (γ=1, β=0) rather than copied from the donor
layer. Copying them costs **+4.12 nats**; resetting them costs **0.000** (max |Δlogit| =
1.3e-3 against a mean logit magnitude of 3.82). Inserting the same block mid-stack costs
+0.23 nats regardless.

A second contribution is methodological: a pattern for pulling **continuous internal-health
telemetry off a live training run at approximately zero marginal GPU cost** (§6.1) — atomic
checkpoints so the run is readable from outside, scoring on the idle device (8.9–22.2 s on CPU
against 13.2–16.6 minutes of exclusive GPU for a downstream evaluation), and streaming the
already-free per-layer signals straight into the training log. On a machine where the GPU *is*
the training run, this is the difference between a live feed and a post-mortem.

We further show that a sixth layer at this scale has no work to do under three independent
initialisations (§5); that single-layer ablation cost over-attributes, with
`marginal(L4|L3) = −1.891`; that every correlational diagnostic we built misled at least once
while causal ablation never did (§6); that annealing corpus dominates annealing schedule by
11× (§7); and that changing only the evaluation mixture moves reported validation loss by
0.225 nats — over half the size of the project's entire pretraining progress (§8).

Total cost: **35.4 GPU-hours**, ~3.0B tokens processed.

---

## 1. Introduction

The constraint that shaped this project was a hard cap of 30M parameters, on a single RTX
3050 Ti Laptop GPU with 4 GB of VRAM. Under such a cap the interesting question is not *how
well can this do* — the answer is known to be "worse than BERT-base" — but *how much of the gap
can be closed by spending the budget better*. We attacked that with architecture surgery
(structured pruning, layer grafting, depth reallocation), data curriculum design,
masking-scheme changes, an alternative pretraining objective, and a battery of diagnostics.

This report is organised around what the experiments established rather than around the
chronology. Checkpoint lineage and the chronological build log are in
[`checkpoints/HISTORY.md`](checkpoints/HISTORY.md) and [`WORKLOG.md`](WORKLOG.md).

---

## 2. Setup

| | v1 line | v2 line (final) |
|---|---|---|
| parameters | 28.733M | **27.952M** |
| layers / heads | 4 / 4 | 6 / 4 |
| `d_model`, `d_k`=`d_v` | 768, 64 | 768, 64 |
| `d_ff`, activation | 3072, ReLU | 1792, GELU |
| embedding | ALBERT-factorised `V×128 → 128→768` | same |
| MLM head | tied to word embeddings | same |
| normalisation | **post-LN** | post-LN |
| sequence length | 128, contiguous packing (no padding, no attention mask) | same |
| masking (train) | BERT 80/10/10 scattered | geometric spans, p=0.2, clipped at 10 |
| masking (eval) | scattered | scattered |

Parameter split of the v2 model: FFN 16.53M (59%), embedding + projection 6.53M (23%),
attention 4.72M (17%), norms 0.02M. Attention holds roughly half the share it does in
BERT-base (~33%): with `n_heads · d_k = 256` against `d_model` 768, each block writes a
rank-≤256 update into a 768-dimensional stream.

**Data.** WikiText-103 119M tokens, UltraFineWeb-en 341M, TinyStories 475M, Cosmopedia 403M,
IMDB 23M — tokenised to packed `uint16` memmaps, contiguous packing.

**Training.** bf16, AdamW, micro-batch 16 × grad-accum 16 (effective batch 256). Measured VRAM
fit fixed the micro-batch: 16 → 1.84 GB / 91 ms; 32 → 3.25 GB / 188 ms; 48 → 4.63 GB / 507 ms
(spills to WSL2 shared memory); 64 → 6.03 GB / 1751 ms.

### 2.4 Evaluation protocol

Every cross-model number here is scored on **one fixed held-out mixture**: the tail 1% of each
Phase-A corpus, token-weighted 55/35/10 (Cosmopedia / UltraFineWeb / TinyStories), identical
RNG seed, so all models see byte-identical masked inputs. Two masking schemes: `scattered`
(BERT 15%, 80/10/10) and `span` (geometric p=0.2, clipped at 10). 15 batches of 16×128; eval
noise ≈ ±0.05 nats.

This protocol was introduced late. Adopting it retroactively converted an entirely
uncomparable set of run logs into a comparable table, and §8 quantifies what its absence had
been costing.

---

## 3. What worked

### 3.1 Data curriculum: the largest single lever in the project

The staged curriculum produced the two largest improvements we recorded, both of them corpus
changes rather than architecture changes.

| checkpoint | data added | scat_loss | Δ |
|---|---|---|---|
| `ckpt` | WikiText-103, 119M | 5.9866 | — |
| `ckpt2` | + UltraFineWeb | 3.0389 | **−2.948** |
| `ckpt3` | + TinyStories + IMDB (replay) | 2.7089 | −0.330 |
| `ckpt_v2` | + Cosmopedia (55% weight), 734M | 2.3280 | **−0.381** |
| `ckpt_v2_anneal_cosmo` | 4k-step Cosmopedia anneal | **2.2738** | −0.054 |

Cosmopedia was weighted highest in Phase A on an explicit argument: it was the only genuinely
*new* data, since `ckpt3` had already seen UltraFineWeb for ~2.5 epochs, near the point where
repeats stop paying. The per-corpus breakdown confirms this paid — Cosmopedia improved by 0.81
nats while everything else moved by a quarter of that.

### 3.2 Replay mixing works, and we have the control

Continued pretraining on a new corpus normally causes catastrophic forgetting. We mixed each
new phase with replay from earlier corpora. IMDB provides an unintentional but clean negative
control: it was dropped **entirely** from the v2 mixture while the other three were retained.

| checkpoint | UltraFineWeb | TinyStories | IMDB | Cosmopedia |
|---|---|---|---|---|
| `ckpt` (WikiText only) | 5.9660 | 5.4163 | 6.0371 | 6.3295 |
| `ckpt2` (+UFW) | 2.8882 | 2.4936 | 3.8784 | 3.4847 |
| `ckpt3` (+TinyStories+IMDB, replay) | 2.9511 | 1.0227 | 2.9255 | 3.1006 |
| `ckpt_v2` (+Cosmopedia, replay) | 3.2132 | 1.2459 | 3.8733 | 2.2875 |
| `ckpt_v2_anneal_cosmo` (final) | 3.3605 | 1.3430 | 3.9433 | 2.1129 |

*Each column is a separate held-out split; columns are comparable only down, not across.*

Reading the `ckpt3 → ckpt_v2` transition against the replay weight:

| corpus | weight in the v2 mixture | Δ loss |
|---|---|---|
| Cosmopedia | 55% (new) | −0.813 |
| UltraFineWeb | 35% | +0.262 |
| TinyStories | 10% | +0.223 |
| **IMDB** | **0%** | **+0.948** |

Any replay at all held forgetting to ≈0.25 nats; zero replay cost 0.95, close to four times as
much — and dropping from 35% to 10% cost nothing measurable. **The replay *floor* matters much
more than the replay *fraction*:** 10% is enough, 0% is not.

The same table also characterises annealing honestly. The Cosmopedia anneal improved
Cosmopedia by 0.175 nats and degraded all three other corpora (UFW +0.147, TinyStories +0.098,
IMDB +0.070). **Annealing specialises; it does not uniformly improve.** This is why the anneal
simultaneously measured −0.054 on the Phase-A mixture and +0.057 on IMDB — an apparent
contradiction at the time.

### 3.3 Span masking

Following SpanBERT, the v2 line trains on geometric span masking (p=0.2, clipped at 10, mean
span 3.80) and evaluates on scattered so the logged accuracy stays comparable with earlier
runs. Span masking removes the ability to infill from immediate neighbours, forcing
representation rather than local autocomplete.

It worked, on two independent readings:

1. **The model improved more on the harder task.** From `ckpt3` to final, scattered accuracy
   rose 4.6pp while span accuracy rose 6.7pp.
2. **The intrinsic difficulty gap narrowed**, from +1.932 nats for the scattered-trained
   `ckpt3` to +1.640 for the span-trained final model — a 0.29-nat narrowing, six times eval
   noise.

That gap is worth stating on its own, because it is a frequent source of false alarms.
Measured on one fixed model with identical data and seed:

| masking scheme (`mlm_prob` 0.15) | mean span | mlm_loss | masked_acc | vs scattered |
|---|---|---|---|---|
| scattered | 1.16 | 2.3655 | 0.5814 | — |
| uniform 2–4 | 2.94 | 3.5002 | 0.4362 | +1.135 |
| geometric p=0.2 | 3.64 | 4.1054 | 0.3662 | **+1.740** |
| geometric p=0.1 | 5.36 | 4.4758 | 0.3186 | +2.110 |

A model that trains on spans and evaluates on scattered shows train and validation curves
~1.7 nats apart **by design**. Anyone reading those logs without this table would diagnose a
catastrophic bug.

### 3.4 WSD with the decay phase left unspent

`ckpt3` used cosine decay to zero and demonstrated it bought nothing: best validation loss
arrived at step 6,400 while the LR was still at 55% of peak, and the fully annealed endpoint
was *worse* than the mid-run best. Every later run used Warmup–Stable–Decay with
`decay_frac = 0`, ending flat.

Two benefits followed. The run stays extendable without a re-warm, and the decay phase becomes
a separate, controllable experiment — which is exactly what made the corpus-controlled
annealing comparison of §7 possible at all. **Spending the decay should be a decision, not a
side effect of the schedule.**

### 3.5 Installing an induction circuit into a masked language model

We coerced the encoder into an autoregressive generator — append three `[MASK]` tokens,
supervise and commit slot 1, slide the window — as a probe of what the MLM had absorbed. Two
capabilities were missing and both were installable in minutes of CPU-class fine-tuning.

- **Termination.** The base model never emits EOS in a generation loop. Oversampling EOS
  positions (30% of supervised slots) during a 400-step continuation fine-tune fixes it.
- **Copy-from-context is absent at baseline: 0.8%.** Induction requires a two-layer circuit —
  a previous-token head feeding a copy head — and four heads per layer leave no room for one to
  form incidentally under a masked objective. A 600-step fine-tune on two-entity templates
  (25% of the batch, the rest replay) raises copy accuracy to **93.3%** on trained-style
  prompts and **80.0%** on 200 entities held out of the training pool, in 6.5 minutes.

The generalisation to held-out entities is what makes this a circuit rather than memorisation.
The result is that the capability was **cheap and absent**: nothing in 2.1B tokens of MLM
pretraining gave the model a reason to build it, and once given a reason it built it in 6.5
minutes.

Two honest caveats. The fine-tune over-corrects at high copy weight — *"We went to Paris last
year with Maria. I loved chatting with"* completes to *"Paris last year with Maria"*. And there
is an open failure: generation loops on prompts shorter than 8 tokens, because training sampled
context length uniformly over [8, 125] while generation *starts* at the prompt length. All four
sub-8-token prompts break; all five longer prompts are clean. Fix identified (log-uniform
context sampling from length 1), not implemented.

### 3.6 The diagnostic tooling

The tools in `tools/` are themselves a result — arguably the most reusable one. Three earned
their cost repeatedly: `span_eval.py` (the fixed-protocol evaluation of §2.4),
`layer_contrib.py` (causal ablation cost, single and pairwise, §5), and `audit_eval.py` (which
mechanically checks the failure modes of §8). Each converted a recurring argument into a number.

More than the individual tools, the *pattern* they share is the contribution: continuous
internal-health telemetry pulled off a live training run for ~zero marginal GPU cost. That is
**§6.1**, and it is the part most directly transferable to anyone training on hardware they
also have to use for something else.

---

## 4. Function-preserving grafts in post-LN transformers

*Reproduce with `python tools/graft_ladder.py --device cuda`.*

### 4.1 The problem

Growing a trained network by adding layers — progressive stacking, bert2BERT, Net2Net — is
meant to be near-free. The standard trick for making it *exactly* free is to zero the new
block's output projections so the block is an identity at step 0.

This is folklore for **pre-LN**, where a block computes `x + f(LN(x))` and `f ≡ 0` genuinely is
the identity. Our model is **post-LN**:

```
h   = LN₁(MHA(x) + x)
out = LN₂(FFN(h) + h)
```

With `W_O = 0` and `W_out = 0` this collapses to `out = LN₂(LN₁(x))` — two extra LayerNorms,
not an identity. We measured what that costs and found the folklore is recoverable in post-LN,
subject to two conditions that are not usually stated.

### 4.2 Measurement

All variants are built from the same trained 4-layer checkpoint (`ckpt3`) by appending **one**
block. Because building a v2-package model also swaps ReLU→GELU, Δ is against the GELU-only
reference so the activation change is factored out.

| variant (one block appended to a trained 4L stack) | scat_loss | Δ vs ref | max \|Δlogit\| |
|---|---|---|---|
| `ckpt3`, 4L, ReLU — the source model | 2.7089 | — | — |
| GELU swap only, still 4L — **the reference** | 2.9776 | 0.000 | — |
| **`W_O`=0, `W_out`=0, LN affines reset to (1,0)** | **2.9776** | **+0.000** | **0.0013** |
| `W_O`=0, `W_out`=0, LN affines **copied from donor** | 7.1016 | +4.124 | 37.7 |
| only `W_out`=0 (attention live), LN copied | 7.7290 | +4.751 | — |
| full copy-stack — *what this project actually did* | 8.5488 | +5.571 | — |

Mean logit magnitude is 3.82, so `max |Δlogit| = 1.3e-3` is float32 round-off (3.3e-4 relative):
the corrected graft is **exactly** function-preserving. The naive graft destroys the model.

### 4.3 Why it works, and where it stops working

A zero-output post-LN block computes `LN(LN(x))`, which with unit affines equals `LN(x)` — a
per-token standardisation of `x`. That is *not* `x`. But LayerNorm is invariant to per-token
affine rescaling,

```
LN(a·x + b) = LN(x)     for scalar a > 0, b
```

and `LN(x)` is itself exactly such a rescaling of `x` (a = 1/σ, b = −μ/σ). So if the **next**
consumer of the stream is another LayerNorm, the standardisation is annihilated and the graft
is invisible.

At the **top** of the stack the only consumer is the encoder's `final_norm`, and the invariance
applies exactly. **Mid-stack**, the next block's attention reads the residual stream *directly*
— attention logits scale as the square of the stream magnitude — while the graft has just
stripped the previous block's learned output scale (γ, β). The invariance no longer applies:

| insertion point of the zero-graft block (5L/3072) | scat_loss | Δ |
|---|---|---|
| nowhere — 4L reference | 2.9776 | — |
| position 0 (before all trained blocks) | 3.0646 | +0.087 |
| position 2 (mid-stack) | 3.2021 | +0.225 |
| position 4 (appended at top) | **2.9776** | **+0.000** |

Position 0 is cheaper than position 2 only because the embedding-plus-positional stream it
re-standardises is less shaped by a learned LayerNorm than a mid-stack activation is.

### 4.4 The counter-intuitive term

The dominant error is **copying the donor block's LayerNorm affines**: +4.12 of the +5.57
total. Every instinct in progressive stacking says to copy everything from a trained layer —
that is the entire argument for stacking over random init. For the LayerNorm affines
specifically, copying is worse than resetting by more than four nats, because γ and β encode
*what scale the next layer expects*, and a newly inserted block inherits the wrong expectation.

### 4.5 Practical rules

- **Post-LN:** zero-init grafting is exact **only at the top of the stack**, and only with
  `W_O = 0`, `W_out = 0`, **and LN affines reset to (1,0)**. Mid-stack, budget ~0.2 nats.
- **Copy weights, reset norms.**
- **Verify with logits, not loss.** Comparing `max |Δlogit|` against the mean logit magnitude
  turns a judgement call into a pass/fail: 1.3e-3 versus 37.7 is not a debate. The verification
  belongs *inside* the tool that performs the graft.
- **A destroyed output does not mean destroyed knowledge.** `ckpt_v2_init` scored 8.26 nats. It
  recovered to 2.71 — its parent's level — within 1,000 steps, and to 2.38 by step 14,200, a
  level `ckpt3` needed ~42,000 cumulative steps to reach. Loss immediately after a graft is a
  very poor proxy for how much transferred.

---

## 5. Depth at 28M: the sixth layer

We grew 4 layers to 6 by cutting `d_ff` 3072→1792 and spending the savings on depth. Layer 5
pulled its weight. **Layer 4 did not.**

### 5.1 Ablation cost: a causal metric

Correlational metrics describe what a layer *looks like*. We wanted what it is *worth*, so we
defined **ablation cost**: zero the layer's output projections (`W_O`, `l2`, `b2`), leaving the
residual path intact, and measure the increase in held-out loss.

| layer | 0 | 1 | 2 | 3 | **4** | 5 |
|---|---|---|---|---|---|---|
| ablation cost (nats) | 1.186 | 1.683 | 1.749 | 1.594 | **0.073** | 0.963 |

Layer 4 is worth 7.6% of what layer 5 is worth, and 1.0% of the stack's total ablation cost,
while holding an identical 12.7% share of the parameters.

### 5.2 Three initialisations, one answer

| initialisation of layer 4 | steps trained | final ablation cost |
|---|---|---|
| stacked copy of a trained layer | 9,600 | ≈ 0 |
| zero-init graft, 5× LR, no weight decay | 4,600 | ≈ 0 |
| standard `N(0, 0.02)`, **rest of the network frozen** | 18,800 | **+0.062** |

The third run is decisive. Freezing embeddings, the MLM head and blocks 0,1,2,3,5 — leaving
3.545M of 27.952M parameters trainable (12.7%) — removes every escape route: the rest of the
network *cannot* route around layer 4, so whatever it can learn, it must. It climbed from 0 to
+0.062 over 18,800 steps and flattened, ~6% of layer 5's contribution.

**Verdict: position, not initialisation.** Transplanting the frozen-trained layer back into the
joint run improved loss by −0.056 nats instantly — and joint training gave most of it back
within 200 steps. The joint optimiser actively *prefers* the configuration in which layer 4
does nothing.

### 5.3 Pairwise marginals change how ablation should be read

`marginal(i|j) = cost({i,j}) − cost({j})`:

| j = | 0 | 1 | 2 | **3** | 5 |
|---|---|---|---|---|---|
| `marginal(4 \| j)` | −0.012 | +0.031 | −0.019 | **−1.891** | +0.005 |

Layer 4 is worth ≈0 given *any* other layer removed, so redundancy is not the explanation —
nothing was covering for it. But at j = 3, **removing the useless layer 4 halves the apparent
cost of removing layer 3** (3.437 → 1.545).

The mechanism is fragility amplification. Ablating layer 3 produces an off-distribution
residual stream; layer 4, trained only on in-distribution input, amplifies that corruption
rather than absorbing it. Layer 3's headline cost of 3.44 nats is therefore ~55% damage routed
through layer 4 and only ~45% its own contribution.

**Rule: single-layer ablation cost over-attributes to layers sitting below a fragile downstream
layer.** For pruning decisions the pairwise marginal is the number wanted, and it can differ
from the single by a factor of two.

---

## 6. Diagnostics: zero-cost telemetry, and what it can and cannot tell you

### 6.1 Zero-cost live diagnostics

On a 4 GB laptop the GPU *is* the training run. Any diagnostic that wants the GPU competes with
the thing it is diagnosing, and the honest measure of model quality — a downstream fine-tune —
costs 8.3 minutes of exclusive GPU, with the full benchmark suite at 13.2–16.6 minutes. At that
price you can afford it perhaps twice per run, which means flying blind for hours and learning
everything in the post-mortem.

The pattern we ended up with gets continuous internal-health telemetry out of a live run at
approximately zero marginal cost. It has three parts, and it is the most transferable piece of
engineering in this project.

**1. Atomic checkpoints make the run readable from outside.** `atomic_save` writes through
`os.replace`, which is atomic on POSIX. An external process can therefore read `last.pt` at any
instant during training and is guaranteed a complete, self-consistent checkpoint — never a torn
write. This is the load-bearing piece: it means diagnostics do not have to live *inside* the
training process, so they cannot slow it down, crash it, or be lost with it.

**2. Score on the idle device.** `span_eval.py`, `attn_health.py` and `ffn_slack.py` default to
CPU precisely so they never contend for VRAM. Measured wall-clock on the final checkpoint:

| diagnostic | device | cost | what it reports |
|---|---|---|---|
| `ffn_slack.py` | CPU | **8.9 s** | per-layer slack, never-positive fraction, utilisation, kurtosis |
| `attn_health.py` | CPU | **14.8 s** | per-head entropy / contribution / offset, out_rank, centred head similarity |
| `layer_contrib.py` | CPU | **22.2 s** | causal ablation cost per layer |
| *SST-2 fine-tune* | *GPU, exclusive* | *8.3 min* | *the actual downstream number* |
| *full benchmark suite* | *GPU, exclusive* | *13.2–16.6 min* | *SST-2 + STS-B, probe + fine-tune* |

The correlational diagnostics are ~60× cheaper than a downstream evaluation and — more
importantly — run on a **different device**, so their marginal cost to the training run is
essentially zero. We ran them on a 20-minute loop for the entire project, accumulating ~100
readings per metric across 35 GPU-hours.

That cadence is what made trends legible. Individual readings are noisy enough that I
repeatedly mis-called layer 4's trajectory from single cycles — "converged", then
"accelerating", then "regression", all wrong. The fix was fitted slopes over multi-cycle
windows, and those windows only exist because each reading cost nine seconds of idle CPU
instead of a quarter-hour of the GPU.

**3. Stream the already-free signals into the training log itself.** `layer_probe` runs one
extra forward pass on the eval batch that is already resident, materialising attention
probabilities only for the named layers, and writes `l4_ent`, `l4_np`, `l5_ent`, `l5_np` into
`metrics.jsonl` next to the loss. The dashboard (`tools/serve_dashboard.py`, stdlib only, no
dependencies) plots them live. Per-layer health stops being a post-mortem you run and becomes a
curve you watch.

This is what made the layer-4 graft legible while it was happening rather than afterwards.
Attention entropy near `log(T) = 4.85` means a head is averaging rather than selecting;
`never_pos` near 1 means the FFN is switched off. Watching `l4_ent` sit at 3.9 said within
minutes — not hours — that the fresh layer was not differentiating.

**The transferable rule: make the run readable from outside (atomic writes), score on whichever
device is idle, and stream into the log only the signals that are already nearly free.** On
constrained hardware this converts "I will find out in the post-mortem" into a live feed for
approximately nothing.

One caveat, which the rest of this section is about. Everything above concerns **delivery**, not
**validity**. Cheap telemetry is not free of interpretation risk — and three of these
near-free metrics were actively misleading until their nulls were measured. The one diagnostic
that never lied is also the one that needs a real forward pass per layer.

### 6.2 Head similarity had a null of ~0.97, not 0

Max pairwise cosine between heads' attention matrices read 0.98 on layer 3 — apparently
duplicate heads. It is not: **an untrained model reads 0.95–0.99 too.** Attention rows are
probability distributions over 128 positions; an undifferentiated head sits near-uniform, and
the shared 1/T floor dominates the cosine. The metric could not distinguish "duplicated heads"
from "untrained heads" — precisely the two cases it existed to separate.

Centring across heads restores a meaningful null: **raw 0.95–0.99, centred −0.23…−0.07** on a
freshly initialised model. The null is *negative* because four centred vectors sum to zero, so
undifferentiated heads are mutually anti-correlated.

A second bug compounded it: zeroing the diagonal before the argmax both won the argmax and
clamped the reported maximum, so the tool reported the nonsense pair `h0~h0` at 0.00. The
diagonal must be `−inf`.

After the fix, layer 3's 0.98 survived (genuinely duplicated) while layers 4 and 5's 0.86 and
0.98 collapsed to 0.18 and 0.27 — untrained, not duplicated. **Every conclusion drawn from the
uncentred metric had to be discarded.**

### 6.3 The same failure one level up: latent-target pretraining

We ran a data2vec/JEPA-style latent-prediction branch (EMA teacher, stop-gradient, narrow
predictor, momentum 0.996→0.999) for 15,258 steps on 500M tokens. Every internal metric was
encouraging.

| step | centred cos | raw cos | constant-predictor baseline | eff. rank | target std |
|---|---|---|---|---|---|
| 200 | 0.0357 | 0.886 | 0.916 | 240.9 | 0.349 |
| 8,600 | 0.6161 | 0.944 | 0.907 | 232.9 | 0.363 |
| 15,200 | **0.6256** | 0.940 | 0.895 | 234.8 | 0.382 |

Note the raw cosine at step 200: **0.886, against a constant-predictor baseline of 0.916.** A
predictor that ignored its input entirely and emitted the dataset mean would have scored
*higher*. Representations are anisotropic; almost any two vectors are similar. The honest
number was 0.0357.

We caught that one. What we did not catch until downstream evaluation is that the *whole class*
of internal metric was uninformative here: centred cosine rose 17×, effective rank held steady,
target std improved monotonically — and the downstream result was indistinguishable from the
far simpler MLM line (§11).

### 6.4 Zero-output layers poison run-level aggregates

FFN-slack and attention-health tools reported `slack = 100%`, `util = 1.00`, `out_rank = nan`
for a freshly grafted layer whose output projection is identically zero. Those rows entered the
run-level means and flipped an automated verdict to "OK" for several monitoring cycles. Any
health metric of the form *"how much of this layer's output matters"* is undefined when the
output is zero by construction.

### 6.5 The general lesson

**Every proxy metric needs its null measured on an untrained model of the same architecture,
and its degenerate cases enumerated, before it is trusted once.** Ablation cost was the only
diagnostic we never had to retract, because it is defined by an intervention on the model
rather than by a statistic of its activations. It is also ~40× more expensive per reading. That
trade was worth it.

---

## 7. Annealing: corpus dominates schedule

Phase A used WSD with `decay_frac = 0`, deliberately leaving the decay phase unspent. We then
spent it twice from the *identical* checkpoint with *identical* LR, schedule, seed and step
count (cosine 2e-4 → 0, no warmup, 4,000 steps), varying only the corpus.

| anneal corpus | own val_loss | held-out result vs parent | outcome |
|---|---|---|---|
| TinyStories | **0.9989** | fell *below* `ckpt3` within 1,800 steps | abandoned |
| Cosmopedia | 2.0487 | −0.054 nats, +1.18pp span acc | kept; final model |

TinyStories produced by far the lowest validation loss the project ever recorded, and it was
pure artefact: TinyStories is deliberately constructed with a ~1,500-word vocabulary at a
3–4-year-old reading level. Annealing on it *narrows the lexicon*; it does not teach fluency.
It lost three quarters of v2's advantage over `ckpt3` in 800 steps. Cosmopedia — same schedule,
synthetic textbook prose — did **11× less damage** and net helped.

Because everything except the corpus was held fixed, this isolates data register as the causal
variable. **Annealing is a data-selection decision, not a schedule trick, and the loss on the
anneal corpus itself carries no information about whether it worked.**

---

## 8. Evaluation validity

A reported validation loss of ~2.04 looked implausibly good — a well-trained 124M GPT-2 on 10B
tokens lands near 3.3. We audited it against the standard failure modes.

**1. It was a different task.** MLM is not autoregressive LM; bidirectional context makes the
token far easier to predict. There was no valid comparison to GPT-2 in the first place. A
category error rather than a bug, but the most consequential item on the list.

**2. The eval mixture, not the model, was doing most of the work.** For the *same* checkpoint:

| evaluation mixture | scat_loss |
|---|---|
| Cosmopedia only (what that run logged) | 2.0487 |
| Phase-A mixture, 55/35/10 (the protocol) | 2.2738 |

**0.225 nats from the evaluation set alone** — over half the size of the entire project's
pretraining improvement (0.435 nats).

**3. Cross-run contamination is real and easy to miss.** The split is contiguous (val = tail
1%), so it is clean *within* a run. But `limit_tokens` changes where the boundary falls, and
weights are inherited across runs:

```
tinystories   n = 73.4M   train=[0, 72.6M)  val=[72.6M, 73.4M)
              prior run   train=[0, 165.3M)  -> VAL INSIDE PRIOR TRAIN: CONTAMINATED
```

v2's TinyStories validation range sits entirely inside `ckpt3`'s training range, and v2
inherited `ckpt3`'s weights. UltraFineWeb was clean; TinyStories was not.

**4. Averaging bias was real but negligible.** Mean-of-batch-means 2.2743 vs token-weighted
2.2760: a −0.0016 nat bias.

**5. Data was not pathologically repetitive.** 0 of 320 validation blocks byte-matched any of
199,989 sampled training blocks; the top-8 scored tokens account for 22.5% of scored positions
across 2,357 distinct types.

**6. The generalisation gap is small but nonzero.** Train-range 2.1427 vs held-out 2.3251:
+0.18 nats, −1.2 accuracy points. Consistent with a capacity-bound, zero-dropout model that is
not overfitting.

---

## 9. Width–depth reallocation

Three measurements on `ckpt3` motivated the v2 architecture:

- **`d_ff` was ~3.5× oversized.** 99% of FFN activation variance fit in 133–866 of 3,072
  directions, depending on layer.
- **Layer 0 had 2,697 of 3,072 units that never fired once** over 2.0M tokens under ReLU.
- **Residual-stream effective rank was still climbing at the last layer** (149 → 269 → 483 →
  523 of 768), so depth had headroom where width did not.

We cut `d_ff` to 1792, swapped ReLU→GELU — scoring the pruning importances *under GELU*, since
scoring under ReLU would discard units whose pre-activations are negative (2,755 in layer 0)
that GELU passes with a small non-zero output — and spent the savings on two extra layers,
landing at 27.952M under the cap.

**The width cut cost almost nothing**, as predicted: structured pruning 3072→1792 cost +0.083
nats on top of the activation swap. After training, `r99/d_ff ≤ 0.56` on every layer — width
was still not the binding constraint even at 1792.

**The depth it funded returned almost nothing**: layer 5 pulled its weight, layer 4 did not
(§5), and the end-to-end gain did not survive to the downstream task (§11).

This is what scaling laws predict. The honest summary: **reallocating a fixed parameter budget
between width and depth is a rounding error compared with changing the budget.**

---

## 10. Engineering results

**Freezing 87% of parameters buys 1.6×, not 8×.** The frozen-stack experiment trained only
layer 4 — 12.7% of parameters — at 0.858 s/step against 1.397 s/step for full joint training:
1.63×. The reason is FLOP structure, not parameter count: the forward pass is unchanged (all
six layers), and the backward pass must still traverse layer 5 to reach layer 4; only the four
layers *below* the trainable one skip backward. Predicted 1.8×, measured 1.63×. **Count the
backward passes you actually skip, not the parameters you froze.**

**Budget by steps, not wall-clock, on a machine that sleeps.** A `--max-seconds 10800` budget
using `time.time()` counted sleep and terminated a run at step 1,851/8,800 after 37 minutes of
real compute.

**A watchdog inside the thing it watches is not a watchdog.** Every recovery mechanism we built
initially lived inside the session that sleep suspended, dying at the same instant as the
training it existed to observe. Moving it to cron covered crashes but not Windows sleep, which
*destroys* the WSL2 VM — confirmed by two distinct boot IDs with an 8h38m hole in
`journalctl --list-boots`. There is no kernel running to fire cron.

**A watchdog that spawns a daemon can disable itself.** The supervisor held its lock on fd 9;
`setsid` children inherit open descriptors, so the long-lived dashboard *it* spawned captured
fd 9 and held the lock for its lifetime. Every subsequent cron tick died at `flock -n 9` and
exited 0, silently, for ~5 hours. The one recovery test that passed did so only because the
dashboard alive at that moment had been started from a shell. **The act of the supervisor
launching a daemon is what disabled the supervisor — that test could not have caught the bug
however carefully it was run.**

**`${VAR:-default}` substitutes on empty, not merely on unset.** `BOOST_LAYERS=""` fell through
to the default and silently launched two runs with a 5× learning rate that had been explicitly
ruled out. `${VAR-default}` is the correct form.

**Grep for a process name and you will match your own shell.** A GPU-wait loop using
`ps | awk '/pretrain_v2/'` matched any command that merely *mentioned* the string, and wedged
forever. Poll the resource (`nvidia-smi` VRAM), not the name.

**Sparse-file apparent size is not disk usage.** Packed `.bin` memmaps are allocated at target
size and written incrementally; the holes cost zero real blocks. Truncating them reduced
apparent size 3.7 GB → 2.4 GB and freed nothing — and requires updating the manifest in
lockstep, since the dataset memmaps with `shape=(target_tokens,)`.

---

## 11. Scoreboard, and the central negative result

### 11.1 Pretraining progressed clearly

| checkpoint | step / arch | scat_loss | scat_acc | span_loss | span_acc |
|---|---|---|---|---|---|
| `ckpt` — WikiText-103 | 8,800 · 4L/3072 | 5.9866 | 0.2531 | 6.9932 | 0.1695 |
| `ckpt2` — +UltraFineWeb | 20,789 · 4L/3072 | 3.0389 | 0.4819 | 4.9318 | 0.2884 |
| `ckpt3` — +TinyStories+IMDB | 12,964 · 4L/3072 | 2.7089 | 0.5295 | 4.6408 | 0.3118 |
| `ckpt_v2_init` — graft, untrained | 0 · 6L/1792 | 8.2635 | 0.0125 | 8.4851 | 0.0075 |
| `ckpt_v2` @ 14.2k | 14,200 · 6L/1792 | 2.3828 | 0.5661 | 4.0408 | 0.3588 |
| `ckpt_v2` @ 18.4k (pre-anneal) | 18,400 · 6L/1792 | 2.3280 | 0.5759 | 3.9809 | 0.3671 |
| `ckpt_v2_l4only` — frozen-stack L4 | 18,800 · 6L/1792 | 2.3136 | 0.5781 | 3.9658 | 0.3695 |
| **`ckpt_v2_anneal_cosmo`** — final | +4,000 · 6L/1792 | **2.2738** | 0.5759 | **3.9135** | **0.3789** |

`ckpt3` → final is **−0.435 nats scattered, −0.727 nats span, +6.7pp span accuracy** — ~9× eval
noise and unambiguously real. The latent-target branch is absent because it never trained an
MLM head; scoring it under MLM reads 11.98 nats and means nothing.

### 11.2 Downstream did not

SST-2 dev is 872 examples: SE ≈ 1.2pp. STS-B dev is 1500: SE ≈ 1.7pp.

| model | SST-2 (standalone) | SST-2 probe | SST-2 ft | STS-B probe | STS-B ft | STS-B 0-shot |
|---|---|---|---|---|---|---|
| `ckpt3` | 0.8567 | 0.7741 | 0.8509 | 0.5971 | 0.4199 | 0.5610 |
| `ckpt_jepa` | 0.8429 | 0.7362 | **0.8647** | 0.6035 | **0.6111** | 0.4600 |
| `ckpt_v2_anneal_cosmo` | **0.8601** | 0.7431 | 0.8509 | 0.5853 | 0.1434 ⚠ | 0.5558 |
| *BERT-base (110M)* | *~0.927* | | | | | |

**This is the project's central negative result.** A −0.435-nat improvement in held-out MLM
loss produced a **0.0pp** change on the SST-2 fine-tune harness and +0.34pp (0.3 SE) on the
standalone harness. Nothing here is distinguishable from noise: the full spread across three
very different pretraining regimes is 0.8509–0.8647, i.e. **1.1 SE**.

Worse for interpretation, **the two SST-2 harnesses disagree by more than any of our
interventions moved the number.** The same final checkpoint reads 0.8601 standalone and 0.8509
in the benchmark — a 0.92pp gap from selection protocol and split handling alone, against a
0.34pp effect size. And the latent-target branch, which the standalone harness ranked last, is
the *best* model on both benchmark fine-tunes.

The honest reading: at 28M parameters on an 872-example dev set, **we built a measurement
instrument too coarse to detect what we were optimising.** Resolving a 1pp difference at this
standard error would need roughly four times more labelled evaluation data. That should have
been checked before the third architecture experiment, not after the last one.

The remaining ~7pp gap to BERT-base is capacity, not allocation. The final model's lineage
processed ~2.18B tokens — 78 tokens per parameter, far past any compute-optimal ratio — while
validation loss plateaued at the eval-noise floor and zero dropout produced no overfitting. All
three signals point the same way.

⚠ `stsb finetune = 0.1434` is broken, not a result: a fine-tune cannot legitimately land four
times below its own frozen probe (0.5853). It read 0.29–0.32 throughout training, i.e. it
diverged. The identical code gave 0.4199 and 0.6111 on other backbones. Undiagnosed.

---

## 12. What we would do differently

1. **Size the evaluation before sizing the experiment.** SE ≈ 1.2pp on SST-2 dev makes any
   intervention worth less than ~2.5pp undetectable. We ran three architecture experiments
   against an instrument that could not resolve their effects.
2. **Measure every proxy metric's null on an untrained model before trusting it once.** The
   head-similarity metric read 0.97 at random init. Ten minutes of work would have caught it;
   instead it invalidated a week of readings.
3. **Prefer causal metrics.** Ablation cost was ~40× more expensive per reading and the only
   diagnostic never retracted. Budget for it.
4. **Use pairwise marginals, not singles, for pruning decisions.** `marginal(4|3) = −1.891`
   means single-layer costs can be twice wrong, in the direction that matters.
5. **Fix the evaluation mixture globally on day one.** Per-run `val_loss` on a per-run mixture
   is not a metric; it is a diary entry.
6. **Keep a replay floor, not a replay fraction.** 10% replay retained as well as 35%; 0% cost
   four times as much.
7. **Do not spend a parameter budget reallocating shape.** At a fixed cap, width–depth trades
   are noise. If the cap is the constraint, the honest experiment is to raise it.
8. **Verify function-preserving transforms with logits, automatically, inside the tool that
   performs them.**

---

## 13. Limitations

Single seed throughout: no result here carries an error bar from repetition, only from
evaluation-set size. MLM evaluation uses 15 batches (~30k tokens, ±0.05 nats).

The §4 graft results are exact at the logit level and do not depend on seed. The §5 layer-4
result replicates across three initialisations and 40+ ablation readings and is the strongest
evidence in the report. The §7 11× ratio is a single pair of runs and should be read as an
order of magnitude, not a coefficient. The replay-retention control in §3.2 is one transition
with four corpora at four weights, and the 0% condition was not deliberately designed as a
control. All downstream numbers are one fine-tune run each, and §11.2 argues they should not be
over-read.

The comparison to BERT-base is from published figures and is not a controlled comparison:
different data, tokenizer, training budget and evaluation harness.

---

## Appendix — reproducing the headline measurements

```bash
# Section 4 -- the graft ladder, end to end (builds every variant, scores it,
# and runs the logit-level function-preservation check)
python tools/graft_ladder.py --device cuda

# fixed cross-model MLM evaluation (the only comparable pretraining numbers)
python tools/span_eval.py --ckpt checkpoints/ckpt_v2_anneal_cosmo/last.pt \
       --baseline checkpoints/ckpt3/last.pt --device cuda

# causal layer value: single + pairwise ablation cost
python tools/layer_contrib.py --full \
       --ckpt checkpoints/ckpt_v2_anneal_cosmo/last.pt

# is the reported val loss measuring what it claims?
python tools/audit_eval.py --device cuda \
       --ckpt checkpoints/ckpt_v2_anneal_cosmo/last.pt \
       --data-name "cosmopedia:403451249:403451249,ultrafineweb_en:256832885:256832885,tinystories:73377468:73377468" \
       --prior-runs "ultrafineweb_en:167000000,tinystories:167000000"

# masking-scheme difficulty offsets
python tools/masking_difficulty.py --ckpt checkpoints/ckpt3/last.pt

# attention health -- head_sim is CENTRED; the raw null is ~0.97
python tools/attn_health.py --ckpt checkpoints/ckpt_v2_anneal_cosmo/last.pt
```

Numeric outputs backing the tables are committed in
[`results/paper_measurements.json`](results/paper_measurements.json) and
[`results/graft_ladder.json`](results/graft_ladder.json).
