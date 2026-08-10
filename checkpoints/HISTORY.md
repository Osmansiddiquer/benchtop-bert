# Checkpoint history

What came from what, what each run was actually testing, and how it ended.
See `NAMING.md` for the naming scheme, and [`../REPORT.md`](../REPORT.md) for the analysis
that these runs feed into.

> **Read the loss numbers carefully.** Each run's `val_loss` is measured on **its own**
> data mixture, so the column is **not comparable across rows**. TinyStories reads ~1.1
> nats easier than the Phase-A mix for the *same model*; the JEPA runs report a latent
> cosine objective, not cross-entropy at all. The only cross-comparable numbers are the
> **downstream results** at the bottom and the `span_eval` figures, which score every
> checkpoint on one fixed held-out mix.

---

## Lineage

```
ckpt  (wikitext-103)
  └── ckpt2  (UltraFineWeb)
        └── ckpt3  ── the baseline everything is measured against
              ├── ckpt_span            span-masked variant
              ├── ckpt_jepa            latent-target pretraining      [FAILED]
              │     └── ckpt_jepa_imdb
              │           └── ckpt_sst2_jepa                          0.8429
              ├── ckpt_v2_init         prune 3072→1792 + GELU + stack to 6L
              │     └── ckpt_v2        Phase A, 733.7M tokens
              │           ├── ckpt_v2_l4only        frozen-stack L4 study
              │           │     └── (layer 4 transplanted back into ckpt_v2)
              │           ├── ckpt_v2_anneal_tiny   TinyStories anneal  [ABANDONED]
              │           └── ckpt_v2_anneal_cosmo  Cosmopedia anneal
              │                 └── ckpt_v2_cosmo_imdb
              │                       └── ckpt_sst2_v2cosmo            0.8601  ← best
              └── ckpt_autoreg/*       MLM → autoregressive fine-tunes
```

---

## The pretraining line

### `ckpt` — phase 1
| | |
|---|---|
| **parent** | scratch |
| **arch** | 4 layers, d_ff 3072, ReLU, d_model 768, d_embed 128 (ALBERT factorised), 28.7M |
| **data** | wikitext-103 |
| **technique** | plain MLM, 15% scattered masking, lr 1e-3 |
| **end** | 8,800 steps, val_loss 2.5435, masked_acc 0.5561 |

The architecture never changed after this point except in the v2 line. Small, clean corpus —
enough to prove the stack trained at all.

### `ckpt2` — phase 2
| | |
|---|---|
| **parent** | `ckpt` |
| **data** | UltraFineWeb (English, quality-filtered web) |
| **technique** | same MLM, lr 5e-4, resumed and extended past its nominal 8,000 steps |
| **end** | step 20,400, val_loss 2.7616, masked_acc 0.5255 |

Loss *rose* against phase 1 because the data got harder — web text vs. curated encyclopedic
prose. Later produced a surprisingly strong SST-2 number (0.8567 via `ckpt_sst2_v3`), which
was never fully explained.

### `ckpt3` — phase 3, **the baseline**
| | |
|---|---|
| **parent** | `ckpt2` |
| **data** | UltraFineWeb 167M + TinyStories 167M + IMDB 91M ≈ 430M tokens |
| **technique** | cosine anneal to zero, lr 3e-4; IMDB folded in to bias toward review text |
| **end** | step 12,964, val_loss 2.1613 (best 2.0557 @ 11,600), masked_acc 0.5945 |

Every later comparison is against this. Notable finding: the cosine decay bought nothing —
best val_loss arrived at step 6,400 while the LR was still at 55% of peak, and the fully
annealed endpoint was *worse* than the mid-run best. That result is why later runs used
WSD with `decay_frac=0`, keeping the decay phase unspent and the run extendable.

### `ckpt_span`
| | |
|---|---|
| **parent** | `ckpt3` (uncertain — no `init_from` recorded in metrics) |
| **data** | UFW 67M + TinyStories 67M + IMDB 37M, i.e. half of ckpt3's budget |
| **technique** | **SpanBERT-style geometric span masking** (p=0.2, clipped at 10, mean span 3.80) instead of scattered |
| **end** | 5,185 steps, val_loss 2.3494, masked_acc 0.5743 — checkpoint since deleted |

Established the **span/scattered difficulty gap: ~1.74 nats**, measured two independent
ways. Span masking removes the ability to infill from immediate neighbours, so it forces
real representation instead of local autocomplete. Every later run trains on spans and
evaluates on scattered, which is why train and val loss sit ~1.7 nats apart by design.

---

## Dead ends

### `ckpt_jepa` — latent-target pretraining **[FAILED]**
| | |
|---|---|
| **parent** | `ckpt3` |
| **data** | Cosmopedia 350M + UltraFineWeb 150M = 500M tokens |
| **technique** | data2vec/JEPA: EMA teacher, stop-gradient, narrow predictor, top-K layer averaging, momentum 0.996→0.999. Predicts **embeddings**, not tokens |
| **end** | 15,258 steps; → `ckpt_jepa_imdb` → `ckpt_sst2_jepa` = **0.8429** SST-2 |

The instructive failure of the project. Every *internal* metric was encouraging — centred
cosine rising, effective rank stable, target std improving — and every one was wrong. The
downstream number came in below the far simpler ckpt3 line.

It also produced a methodological catch worth keeping: the raw cosine between predictor and
target read 0.884, which looks excellent until you compute the constant-predictor baseline —
**0.898**. Representations are anisotropic; almost any two vectors are similar. The honest
number after centring was **0.0357**.

Its one genuine win was STS-B fine-tune at **0.6111**, still the best of any model here.

### `ckpt_v2_anneal_tiny` — TinyStories anneal **[ABANDONED at step 1,800/4,000]**
| | |
|---|---|
| **parent** | `ckpt_v2/pre_anneal.pt` |
| **technique** | cosine 2e-4 → 0, no warmup, TinyStories only |
| **end** | fell **below ckpt3** on scattered accuracy within 1,800 steps |

Its own val_loss looked superb (0.9989) — TinyStories is simply an easier test. Scored on the
common Phase-A mix it lost three quarters of v2's entire advantage over ckpt3 in 800 steps.

Cause: TinyStories was deliberately built with a **~1,500-word vocabulary at a 3–4-year-old
reading level**. Annealing on it narrows the lexicon rather than teaching fluency. The
identical schedule on Cosmopedia did **11× less damage**, which isolates the corpus as the
variable. Kept on disk as the negative control.

---

## The v2 line

### `ckpt_v2_init.pt` — the architecture transfer
| | |
|---|---|
| **parent** | `ckpt3` |
| **arch** | **6 layers, d_ff 1792, GELU**, 28.0M params (under the 30M cap) |
| **technique** | three transfers at once — see below |

1. **FFN pruned 3072 → 1792** by structured importance, scored **under GELU, not ReLU**.
   Scoring under ReLU would have discarded units whose pre-activations are negative — 2,755
   of them in layer 0 — but GELU gives those a non-zero output, so some deserved to live.
2. **Layers 4–5 stacked** from trained layers 2–3 (progressive stacking / bert2BERT).
3. Embeddings, norms and MLM head copied unchanged.

Motivated by three measurements on ckpt3: d_ff was ~3.5× oversized (99% of activation
variance fit in 133–866 directions), layer 0 had 2,697/3,072 ReLU units that never fired over
2.0M tokens, and residual-stream rank was still climbing at the last layer (149 → 269 → 483 →
523 of 768) — so depth had room, width didn't.

> **Post-hoc (Aug 2026): the stacking step was done wrong, and we measured by how much.**
> Scored on the fixed held-out mix, `ckpt_v2_init` reads **8.2635** against ckpt3's 2.7089.
> Decomposing that (see [`../REPORT.md`](../REPORT.md) §2), the width cut is innocent — the
> ReLU→GELU swap costs +0.269 nats and the 3072→1792 prune another +0.083. **Appending the
> layers costs +5.57.** And it did not have to: a block with `W_O = 0`, `W_out = 0` **and its
> LayerNorm affines reset to (γ=1, β=0)** appended at the top of a post-LN stack is *exactly*
> function-preserving (max |Δlogit| 1.4e-3 vs mean logit 3.83, i.e. float32 round-off).
> Copying the donor's LN affines instead — which is what "stacking" means and what this run
> did — accounts for **+4.12 of the +5.57**. Inserting mid-stack costs +0.23 regardless.
>
> The run recovered anyway: 8.26 → 2.38 in 14,200 steps, reaching a level ckpt3 needed ~42,000
> cumulative steps for. So the graft preserved a great deal even while scrambling the output —
> which is itself the lesson: **loss immediately after a graft is a poor proxy for how much
> transferred.**

### `ckpt_v2` — Phase A
| | |
|---|---|
| **parent** | `ckpt_v2_init.pt` |
| **data** | Cosmopedia 403M (55%) + UltraFineWeb 257M (35%) + TinyStories 73M (10%) = 733.7M tokens |
| **technique** | WSD, warmup 0.10, **`decay_frac=0`** so it ends flat and the anneal is a separate phase; geometric span masking, scattered eval |
| **end** | stopped at step 18,400 of 22,389; `pre_anneal.pt` |

Best of the joint-trained line: **−0.39 scat_loss / +4.2pp scat_acc / −0.67 span_loss /
+5.8pp span_acc** against ckpt3 on the common mix.

Cosmopedia was weighted highest because it was the only genuinely *new* data — ckpt3 had
already seen UFW for ~2.5 epochs, near the point where repeats stop paying.

**Intermediate artifacts:** `pre_transplant.pt` (step 14,200, the joint baseline the
layer-4 transplant is measured against), `ckpt_v2_init.pt`.

### `ckpt_v2_l4only` — the layer-4 study
| | |
|---|---|
| **parent** | `ckpt_v2` @ 14,200, with layer 4 re-initialised to N(0, 0.02) |
| **technique** | **every parameter frozen except layer 4** — embeddings, MLM head, and blocks 0,1,2,3,5. 3.545M of 27.952M params trainable (12.7%) |
| **end** | 18,800 steps; layer 4 reached ablation cost **+0.062 nats** |

The cleanest experiment of the project. Layer 4 was measured worth **−0.003 nats** under
joint training — removing it entirely cost nothing. Three hypotheses were tested and killed:

- **Initialisation?** No. Stacked copy (9,600 steps), zero-init graft with 5× LR and no
  weight decay (4,600 steps), and standard random init all converged to ~0.
- **Redundancy?** No. Pairwise ablation, `marginal(4|j) = cost({4,j}) − cost({j})`, stayed at
  zero for every other layer *j* — nothing was covering for it.
- **Routed around?** Partly. Freezing the stack removed the escape route and layer 4 did
  climb from 0 to +0.062 — but that is ~6% of what layer 5 contributes.

Transplanting the trained layer back into the joint run improved loss by **−0.056 nats**
instantly, then joint training gave most of it back within 200 steps.

**Verdict: position, not initialisation.** A 6th layer at that depth and scale has no work
available that L1/L2/L5's existing redundancy doesn't already absorb. It holds 12.7% of the
parameters and delivers ~1% of the stack's total ablation cost (7.6% of what layer 5 gives).

### `ckpt_v2_anneal_cosmo` — the anneal that worked
| | |
|---|---|
| **parent** | `ckpt_v2/pre_anneal.pt` |
| **data** | Cosmopedia only (a second epoch — it was fully consumed in Phase A) |
| **technique** | cosine 2e-4 → 0, **no warmup** — this *is* the WSD decay phase Phase A left unspent |
| **end** | 4,000 steps; **−0.054 scat_loss / +1.18pp span_acc** vs `pre_anneal` on the common mix |

The payoff arrived in the last third, as cosine anneals do. Mildly *worse* on IMDB (+0.057
nats), which is why the downstream numbers were needed to settle it.

### `ckpt_v2_cosmo_imdb` → `ckpt_sst2_v2cosmo` — the current best
| | |
|---|---|
| **parent** | `ckpt_v2_anneal_cosmo` |
| **data** | IMDB 70% / UFW 20% / Cosmopedia 10% |
| **technique** | cosine 5e-5 → 0, 1,500 steps, domain-adaptive pretraining before the task |
| **end** | **SST-2 = 0.8601 ± 0.0117** — best of the project |

Its own val_loss (3.2359) is high because it is scored on an IMDB-weighted mixture the model
had barely seen; that number says nothing about quality relative to other rows.

---

## `ckpt_autoreg/` — MLM → autoregressive

| | |
|---|---|
| **parents** | `ckpt3`, `ckpt_v2/pre_anneal`, `ckpt_v2_anneal_cosmo` |
| **technique** | two phases: (1) next-token prediction expressed through the mask interface — append 3 `[MASK]`, supervise only slot 1, randomised context length, EOS oversampling; (2) same plus 25% two-entity induction templates |

Not a capability line — a probe of what the encoder absorbed. Established that the model
cannot copy from context (baseline ~1%) because induction needs a two-layer circuit and 4
heads per layer leave no room for one to form incidentally.

Known open issue: **looping at short contexts**. Training samples context length uniformly
over [8, 125], but generation *starts* at the prompt length — every sub-8-token prompt begins
in a regime the fine-tune never covered, and all four such prompts break while all five
longer ones are clean.

---

## Downstream results — the only cross-comparable table

| model | SST-2 (standalone) | SST-2 probe | SST-2 ft | STS-B probe | STS-B ft | STS-B 0-shot |
|---|---|---|---|---|---|---|
| ckpt3 | — | 0.7741 | 0.8509 | 0.5971 | 0.4199 | 0.5610 |
| jepa | 0.8429 | 0.7362 | 0.8647 | 0.6035 | **0.6111** | 0.4600 |
| v3 (ckpt2 line) | 0.8567 | — | — | — | — | — |
| v2 (early) | 0.8188 | — | — | — | — | — |
| **v2cosmo** | **0.8601** | 0.7431 | 0.8509 | 0.5853 | 0.1434 ⚠ | 0.5558 |
| *BERT-base* | *~0.927* | | | | | |

⚠ `stsb finetune = 0.1434` for v2cosmo is **broken, not a result** — a fine-tune cannot
legitimately land 4× below its own frozen probe (0.5853). It read 0.29–0.32 throughout
training, so it diverged. The identical code gave 0.4199 and 0.6111 on other models.

Note also that the two SST-2 harnesses disagree for v2cosmo: 0.8601 standalone vs 0.8509 in
the benchmark. Different selection protocol and split handling. The honest reading is
**0.85–0.86**, at the top of our range but not cleanly ahead of JEPA's 0.8647.

---

## What the whole sequence established

**Architecture reallocation is nearly free but nearly worthless.** d_ff 1792 measured no
worse than 3072 on every proxy — FFN slack, utilisation, and attention behaviour was
*identical* between the two — so the width cut cost nothing. The depth it funded contributed
~nothing. Consistent with scaling laws: performance depends on parameters, data and compute,
only weakly on shape.

**Data quality and register dominate annealing.** Same LR, schedule, seed and checkpoint;
only the corpus differed, and the outcomes differed by 11×.

**Correlational diagnostics mislead; ablation doesn't.** never-positive fraction, effective
rank, and head similarity all pointed at problems that ablation showed weren't there — and
head similarity in particular had a null of ~0.97 rather than 0 until it was centred across
heads, because untrained heads all sit at near-uniform attention and are therefore trivially
similar to each other.

**Function-preserving grafts are achievable in post-LN — we just didn't do it.** Reset the new
block's LayerNorm affines and append at the top: exactly zero cost. Copy them: +4.12 nats.
See [`../REPORT.md`](../REPORT.md) §2.

**None of it survived to the downstream task.** ckpt3 → the final model is −0.435 nats
held-out MLM loss (~9× eval noise, unambiguously real) and 0.0pp on the SST-2 fine-tune
harness. The full spread across three very different pretraining regimes is 1.1 SE, and the
two SST-2 harnesses disagree with each other by more than any intervention moved the number.
**The evaluation was too coarse to measure what we were optimising.**

**The 30M parameter cap was always the binding constraint.** The remaining ~7pp gap to
BERT-base is capacity, not allocation.
