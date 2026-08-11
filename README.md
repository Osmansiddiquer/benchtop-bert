# benchtop-bert — a from-scratch BERT-style encoder (laptop-scale)

An encoder-only transformer — attention, blocks, embeddings, MLM head — written from scratch
and pretrained on a single 4 GB laptop GPU (RTX 3050 Ti), under a hard **30M parameter cap**,
then fine-tuned for sentiment. Every component is hand-rolled; only the tokenizer is borrowed
(`allenai/OLMo-1B-hf`, plus an added `[MASK]`).

*Benchtop* as in benchtop instrument: small enough to sit on a desk, built to be measured with.

**Status: closed.** Final model `ckpt_v2_anneal_cosmo` — 27.952M params, 6 layers, `d_ff` 1792,
GELU. **SST-2 dev 85–86%** (BERT-base ≈ 92.7% at 110M params). Total cost: **35.4 GPU-hours**,
~3.0B tokens processed.

> 📄 **[`paper/report.pdf`](paper/report.pdf) — the technical report** (15 pages, LaTeX source
> in [`paper/report.tex`](paper/report.tex)); [`REPORT.md`](REPORT.md) is the same content in
> Markdown. Read either first if you want the findings rather than the code.
>
> 📓 [`checkpoints/HISTORY.md`](checkpoints/HISTORY.md) — checkpoint lineage: what came from
> what, what each branch tested, how it ended.
> 📋 [`WORKLOG.md`](WORKLOG.md) — chronological build log and infrastructure incidents.

---

## The toolkit

The durable output of this project is not the model — it is the instruments built to
interrogate it. All but the first two run **on the CPU in under 25 seconds**, against a live
training job, on a loop, without costing it anything.

| tool | what it measures | what it established |
|---|---|---|
| `graft_ladder.py` | whether a layer graft preserves the function, at the logit level | reset the donor's LN affines: **+4.12 → 0.000 nats** |
| `layer_contrib.py` | **causal** ablation cost per layer, single and pairwise | layer 4 worth 0.073 vs layer 5's 0.963; `marginal(4\|3) = −1.891` |
| `ffn_rank.py` | activation and weight spectra, effective rank | `d_ff` 3.5× oversized ⇒ the entire v2 architecture |
| `dead_neuron_check.py` | per-unit firing rates over 2.0M tokens | 2,697/3,072 layer-0 units dead ⇒ GELU swap, prune scored under GELU |
| `attn_health.py` | per-head entropy / contribution, **centred** head similarity | the raw metric's null is 0.97, not 0 |
| `ffn_slack.py` | slack, never-positive fraction, utilisation | live per-layer health; dead-zone mass vs real width pressure |
| `span_eval.py` | fixed-protocol held-out MLM loss, both maskings | the only cross-run comparable numbers here |
| `audit_eval.py` | train/val overlap, duplicate blocks, averaging bias | found a genuinely contaminated corpus |
| `masking_difficulty.py` | scattered vs span offsets on one fixed model | +1.74 nats — the train/eval gap that looks like a bug and isn't |

## What we set out to do, and what happened

**Goal:** buy downstream accuracy with architecture and data interventions rather than
parameters, under a fixed 30M cap.

**What worked.** The data curriculum was the largest lever in the project (−2.95 nats from
WikiText → UltraFineWeb alone). Replay mixing demonstrably prevents forgetting, with a clean
negative control: the one corpus dropped to 0% replay lost **0.95 nats** while every replayed
corpus lost ~0.25. Span masking paid off on both readings — span accuracy rose 6.7pp against
scattered's 4.6pp, and the intrinsic difficulty gap between the two schemes narrowed by 0.29
nats. And the autoregressive probe installed a working induction circuit into a masked model,
taking copy-from-context from **0.8% → 93%** (80% on held-out entities) in 6.5 minutes.

**What didn't: the headline goal.** Pretraining improved clearly and measurably —
**−0.435 nats held-out MLM loss, +6.7pp span accuracy** from the `ckpt3` baseline to the final
model, ~9× eval noise. Downstream, that bought **0.0pp** on the SST-2 fine-tune harness
and **+0.34pp** on the standalone one, against a standard error of **1.2pp** on an
872-example dev set.

At this scale we built a measurement instrument too coarse to detect what we were optimising.
Pretraining quality improved; the evaluation could not see it.
The full spread across three very different pretraining regimes (plain MLM, span-masked +
re-architected, latent-prediction/JEPA) is 0.8509–0.8647 on SST-2 — **1.1 SE**. The two SST-2
harnesses disagree with each other by 0.92pp, more than any intervention moved the number.

The remaining ~7pp gap to BERT-base is **capacity**, not allocation.

## Findings worth carrying forward

Each is measured, not asserted; §-numbers point into the report.

- **Zero-cost live diagnostics.** Atomic checkpoints (`os.replace`) make a running job
  readable from outside; scoring on the idle CPU costs **8.9–22.2 s** against **13.2–16.6 min
  of exclusive GPU** for a downstream eval; and the already-free per-layer signals stream
  straight into `metrics.jsonl` for live plotting. ~60× cheaper, on a different device, so the
  marginal cost to training is ~zero — which is what made a 20-minute monitoring cadence
  affordable across 35 GPU-hours. §6.1
- **Replay floor, not replay fraction.** 10% replay retained as well as 35%; **0% cost four
  times as much** (0.95 nats vs ~0.25). The same measurement shows annealing *specialises*
  rather than uniformly improving: the Cosmopedia anneal gained 0.175 nats on Cosmopedia and
  lost 0.07–0.15 on all three other corpora. §3.2
- **Span masking transferred.** Training on geometric spans while evaluating on scattered
  improved the harder task more (+6.7pp span vs +4.6pp scattered) and narrowed the intrinsic
  gap between the schemes by 0.29 nats. §3.3
- **An induction circuit is cheap and absent.** Copy-from-context is 0.8% after 2.1B tokens of
  MLM pretraining and 93% after a 600-step fine-tune (80% on held-out entities, 6.5 minutes).
  Four heads per layer leave no room for the two-layer circuit to form incidentally. §3.5
- **Zero-init layer grafting *is* function-preserving in post-LN — with two conditions nobody
  states.** It must be appended at the **top** of the stack (mid-stack costs +0.23 nats no
  matter what), and the new block's LayerNorm affines must be **reset to (γ=1, β=0), not
  copied from the donor**. Copying them costs **+4.12 nats**; resetting them costs **0.000**
  (max |Δlogit| 1.3e-3 against mean logit 3.82). LayerNorm's invariance to per-token affine
  rescaling is what makes the top-of-stack case exact. §4
- **A 6th layer at 28M params has no work to do.** Three independent initialisations — stacked
  copy, zero-init graft with 5× LR, and standard random with the *entire rest of the network
  frozen* — all converged to ≈0 ablation cost. It holds 12.7% of the parameters and returns
  ~1% of the stack's total ablation cost. Cause: position, not initialisation. §5
- **Single-layer ablation cost over-attributes.** `marginal(L4|L3) = −1.891` — removing the
  useless layer 4 *halves* the apparent cost of removing layer 3, because layer 3's damage was
  being amplified downstream rather than absorbed. Use pairwise marginals for pruning
  decisions. §5.3
- **Cheap correlational metrics designed the architecture — after they were calibrated.**
  Effective rank is the win: FFN activations needed only 133–866 of 3,072 directions while
  residual-stream rank was *still climbing* at the last layer, so width was cut and depth
  bought; re-measuring the trained result gave `r99/d_ff ≤ 0.56`, confirming the cut was safe.
  Dead-unit counting (2,697/3,072 layer-0 units never fired) changed *how* the prune was
  scored. But each metric needed its null and degenerate cases audited first: head similarity
  reads **~0.97 at random init**, not 0. **Measure a metric's null on an untrained model before
  trusting one reading.** §6.2–6.3
- **In annealing, corpus register dominates schedule by 11×.** Identical LR, seed, schedule and
  parent checkpoint; TinyStories vs Cosmopedia. TinyStories produced the project's *lowest*
  val loss (0.9989) and was pure artefact — its deliberate ~1,500-word vocabulary narrows the
  lexicon instead of teaching fluency, and it destroyed three quarters of the model's advantage
  in 800 steps. §7
- **Cross-run loss comparison is almost always invalid.** For one fixed model, changing only
  the *eval mixture* moved reported val loss by **0.225 nats** — over half the size of the
  entire project's pretraining progress. Also found: a genuinely contaminated corpus (v2's
  TinyStories val range sits inside `ckpt3`'s train range), and a **+1.74 nat** train/eval
  offset that is purely a masking-scheme difference. §8
- **Width→depth reallocation was free and worthless.** `d_ff` 3072 was ~3.5× oversized (99% of
  activation variance in 133–866 directions; 2,697/3,072 layer-0 units never fired), so cutting
  it to 1792 cost +0.083 nats — and the depth it funded returned nothing. Consistent with
  scaling laws: shape matters weakly; the budget matters. §9
- **Freezing 87% of parameters buys 1.6×, not 8×.** Forward is unchanged and backward must
  still traverse the layers above the trainable one. Count skipped backward passes, not
  parameters. §10

---

## Layout

```
mini_enc_transformer/       the package (hyphens are illegal in import names)
  model/                    v1 architecture: 4L, d_ff 3072, ReLU
  model_v2/                 v2 architecture: 6L, d_ff 1792, GELU
  arch.py                   detect a checkpoint's architecture from its tensors
  data/    dataset.py       memmap-backed blocks + MixtureDataset (replay mixing)
           prep.py          tokenise a corpus -> packed uint16 memmap (resumable)
  training/ pretrain.py     v1 MLM pretraining loop (bf16, warmup->cosine, resumable)
            pretrain_v2.py  v2 loop: WSD, span masking, layer freezing, boost groups,
                            per-layer probes streamed to metrics.jsonl
            jepa.py         latent-target (data2vec-style) pretraining      [dead end]
            finetune_sst2.py  SST-2 fine-tune (mean-pool, LLRD, clean eval split)
            continuation.py / induction.py   MLM -> autoregressive probes
  evaluation/ benchmark.py  probe/finetune x SST-2/STS-B
tools/     graft_ladder.py  reproduces the paper's main result (post-LN graft ladder)
           span_eval.py     THE cross-comparable eval: fixed held-out mix, both maskings
           layer_contrib.py causal layer value: single + pairwise ablation cost
           audit_eval.py    is the reported val loss measuring what it claims?
           build_v2_from_ckpt3.py  prune + stack + activation swap (graft variants)
           graft_l4.py / transplant_layer.py / restack_layers.py   surgery
           attn_health.py / ffn_slack.py / ffn_rank*.py / dead_neuron_check.py
           masking_difficulty.py   scattered vs span offsets on one fixed model
           serve_dashboard.py      live training dashboard (stdlib only)
scripts/                    shell entry points (phase launches, anneal branches)
notebooks/                  pretraining, finetune_sst2, autoreg (generation)
checkpoints/                HISTORY.md, NAMING.md, and the runs themselves
paper/                      report.tex + report.pdf (the technical report)
results/                    downstream JSON + the paper's measurement tables
data/                       packed memmaps + manifests (gitignored, ~2.4 GB)
logs/                       training / prep / diagnostic logs and jsonl histories
```

Everything runs from the repo root — including the notebooks, whose paths are relative to it
rather than to `notebooks/`.

## Quickstart

```bash
# 1. tokenise a corpus into a packed memmap
python -m mini_enc_transformer.data.prep --dataset roneneldan/TinyStories \
       --out data --name tinystories --target-tokens 500000000

# 2. pretrain. --data-name accepts a replay mixture "name:weight:limit"
python -m mini_enc_transformer.training.pretrain_v2 --data-dir data \
       --data-name "cosmopedia:0.55,ultrafineweb_en:0.35,tinystories:0.10" \
       --n-layers 6 --d-ff 1792 --micro-batch 16 --grad-accum 16 \
       --lr 2e-4 --schedule wsd --warmup-frac 0.10 --decay-frac 0 \
       --mask-span-dist geometric --mask-geom-p 0.2 --mask-span-max 10 \
       --eval-span-min 1 --eval-span-max 1 --probe-layers 4,5 \
       --ckpt-dir checkpoints/ckpt_v2 --device cuda

# 3. fine-tune on SST-2 (architecture is auto-detected from the checkpoint)
python -m mini_enc_transformer.training.finetune_sst2 \
       --ckpt checkpoints/ckpt_v2_anneal_cosmo/last.pt \
       --run-dir checkpoints/ckpt_sst2_v2cosmo \
       --out results/finetune_sst2_results_v2cosmo.json

# 4. compare two checkpoints on the FIXED held-out mix (the only valid comparison)
python tools/span_eval.py --ckpt checkpoints/ckpt_v2_anneal_cosmo/last.pt \
       --baseline checkpoints/ckpt3/last.pt --device cuda

# 5. watch it live
python tools/serve_dashboard.py --port 8000      # http://localhost:8000
```

## The model

| | v1 line | v2 line (final) |
|---|---|---|
| params | 28.733M | **27.952M** |
| layers / heads | 4 / 4 | 6 / 4 |
| `d_model`, `d_k`=`d_v` | 768, 64 | 768, 64 |
| `d_ff`, activation | 3072, ReLU | 1792, GELU |
| embedding | ALBERT factorised `V×128` then `128→768` (38.6M → 6.5M) | same |
| MLM head | tied to the word-embedding table | same |
| normalisation | **post-LN** | post-LN |
| sequence length | 128, contiguous packing (no padding, so no attention mask) | same |
| masking (train) | BERT 80/10/10 scattered | geometric spans, p=0.2, clipped at 10 |
| masking (eval) | scattered | scattered — **+1.74 nats easier than train, by design** |

Parameter split: FFN 59%, embedding + projection 23%, attention 17%, norms 0.1%. Attention
holds roughly half the share it does in BERT-base — `n_heads · d_k = 256` writes a rank-≤256
update into a 768-dimensional stream.

No `[CLS]`: OLMo has none and NSP was dropped, so downstream classification **mean-pools** the
encoder hidden states (Sentence-BERT finds this often beats `[CLS]` anyway).

## Training history

Scored on **one fixed held-out mixture**, identical masks and seed — the only comparable
column in this repo. Per-run `val_loss` in `metrics.jsonl` is **not** comparable across rows.

| checkpoint | data | scat_loss | span_acc |
|---|---|---|---|
| `ckpt` | WikiText-103, 119M | 5.9866 | 0.1695 |
| `ckpt2` | + UltraFineWeb | 3.0389 | 0.2884 |
| `ckpt3` — baseline | + TinyStories + IMDB | 2.7089 | 0.3118 |
| `ckpt_v2_init` | prune + stack graft, untrained | 8.2635 | 0.0075 |
| `ckpt_v2` | Cosmopedia/UFW/TinyStories, 734M, WSD | 2.3280 | 0.3671 |
| **`ckpt_v2_anneal_cosmo`** | + 4k-step Cosmopedia anneal | **2.2738** | **0.3789** |

| downstream | SST-2 (standalone) | SST-2 ft | STS-B probe | STS-B ft |
|---|---|---|---|---|
| `ckpt3` | 0.8567 | 0.8509 | 0.5971 | 0.4199 |
| `ckpt_jepa` | 0.8429 | **0.8647** | 0.6035 | **0.6111** |
| `ckpt_v2_anneal_cosmo` | **0.8601** | 0.8509 | 0.5853 | 0.1434 ⚠ |
| *BERT-base (110M)* | *~0.927* | | | |

SE ≈ 1.2pp on SST-2 (872 dev examples), ≈ 1.7pp on STS-B (1500). None of these differences is
significant. ⚠ `stsb finetune = 0.1434` is broken, not a result — it landed 4× below its own
frozen probe and read 0.29–0.32 throughout training. Undiagnosed.

## Things worth knowing before you touch this

- **The model is capacity-bound.** MLM val loss plateaued at the eval-noise floor, zero dropout
  produced no overfitting (held-out gap +0.18 nats, −1.2 acc points), and the final model's
  lineage processed ~2.18B tokens — **78 tokens per parameter**, far past any compute-optimal
  ratio. More parameters will help; more tokens will not.
- **Do not compare `val_loss` across runs.** Each run evaluates on its own mixture. Use
  `tools/span_eval.py`. See §8 of the report for what this costs you if you don't (0.225 nats
  from the eval set alone) and for a real contamination case in this repo.
- **Train and eval use different masking schemes** and therefore sit ~1.7 nats apart by design.
  This is not a bug. `tools/masking_difficulty.py` prints the offset table.
- **`head_sim` in `attn_health.py` is centred across heads.** The raw metric has a null of
  ~0.97. Anything computed from the uncentred value is meaningless.
- **WSD with `decay_frac=0`**, not cosine-to-zero. `ckpt3` showed the cosine decay bought
  nothing — best val loss arrived at step 6,400 while the LR was still at 55% of peak, and the
  fully annealed endpoint was *worse* than the mid-run best. Leaving the decay unspent keeps
  the run extendable and makes the anneal a separate, controllable phase.
- **Model selection never touches the reported set.** The fine-tune carves a selection split
  out of train and evaluates the official 872-example dev set exactly once. Reporting the max
  over many evals on 872 examples is optimistic by 1–2 SE.
- **Budget by steps, not wall-clock** — a sleeping laptop must not spend its training budget
  while suspended. On WSL2, Windows sleep *destroys* the VM; no Linux-side watchdog can survive
  it.
- **The MLM is not a generator.** `notebooks/autoreg.ipynb` coerces it into one (3 trailing
  `[MASK]`s, commit slot 1, slide the window), then fixes never-terminating and never-copying
  with ~10 minutes of fine-tuning. Copy accuracy goes 0.8% → 93%. **Known open bug:** it loops
  on prompts shorter than 8 tokens, because training sampled context length uniformly over
  [8, 125] while generation starts at the prompt length.

## Known-broken / open

- `stsb finetune` diverges for `ckpt_v2_cosmo_imdb` (0.1434, below its own probe). Same code
  works on other backbones.
- The two SST-2 harnesses (`finetune_sst2.py` standalone vs `evaluation/benchmark.py`) disagree
  by ~0.9pp on the same checkpoint. Not reconciled.
- Autoregressive looping at sub-8-token prompts. Fix identified (log-uniform context sampling
  from length 1), not implemented.
- `ckpt_v2_anneal_tiny` and `ckpt_v2_cosmo_imdb` `.pt` weights were deleted during cleanup;
  their numbers survive only in `HISTORY.md` and `results/`.
