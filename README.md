# transformer-v1 — a from-scratch BERT-style encoder (laptop-scale)

A 28.7M-parameter encoder-only transformer — attention, blocks, embeddings, MLM head —
written from scratch and pretrained on a single 4 GB laptop GPU (RTX 3050 Ti), then
fine-tuned for sentiment. Every component is hand-rolled; only the tokenizer is borrowed
(`allenai/OLMo-1B-hf`, plus an added `[MASK]`).

**Best result: 85.7% ± 1.2% on SST-2 dev** (BERT-base ≈ 92.7%, at 110M params).

## Layout

```
mini_enc_transformer/       the package (hyphens are illegal in import names)
  model/                    attention, optimized_mha, blocks, embeddings, encoder,
                            ffn, normalization, positional_encodings, mlm, utils
  data/    dataset.py       memmap-backed blocks + MixtureDataset (replay mixing)
           prep.py          tokenise a corpus -> packed uint16 memmap (resumable)
  training/ pretrain.py     MLM pretraining loop (bf16, warmup->cosine, resumable)
            finetune_sst2.py  SST-2 fine-tune (mean-pool, LLRD, clean eval split)
            continuation.py   teach the MLM to continue text and to stop (WSD)
            induction.py      teach copy-from-context (two-entity templates)
scripts/                    shell entry points (phase launches, cron supervisor)
tools/     serve_dashboard.py  live training dashboard (stdlib only)
           cmp_autoreg.py      before/after generation comparison
notebooks/                  pretraining, finetune_sst2, autoreg (generation), scratch
checkpoints/                ckpt (phase 1), ckpt2 (phase 2), ckpt3 (phase 3),
                            ckpt_autoreg/ (generation branches), ckpt_sst2_v*/ (fine-tunes)
results/                    finetune_sst2_results*.json
data/                       packed memmaps + manifests (gitignored, ~6 GB)
datasets/                   HF datasets on disk (sst2)
logs/                       training / prep / dashboard logs
```

Everything runs from the repo root — including the notebooks, whose paths are relative
to it rather than to `notebooks/`.

## Quickstart

```bash
# 1. tokenise a corpus into a packed memmap
python -m mini_enc_transformer.data.prep --dataset roneneldan/TinyStories \
       --out data --name tinystories --target-tokens 500000000

# 2. pretrain (or continue). --data-name accepts a replay mixture "name:weight:limit"
python -m mini_enc_transformer.training.pretrain --data-dir data \
       --data-name "ultrafineweb_en:0.5,tinystories:0.5" \
       --micro-batch 16 --grad-accum 16 --lr 3e-4 --max-steps 12964 \
       --ckpt-dir checkpoints/ckpt3 --device cuda \
       --init-from checkpoints/ckpt2/last.pt

# 3. fine-tune on SST-2
python -m mini_enc_transformer.training.finetune_sst2 --ckpt checkpoints/ckpt3/last.pt \
       --run-dir checkpoints/ckpt_sst2_v3 --out results/finetune_sst2_results_v3.json

# 4. watch it live
python tools/serve_dashboard.py --port 8000      # http://localhost:8000
```

## The model

| | |
|---|---|
| params | 28.7M (22.2M non-embedding) |
| layers / heads | 4 / 4, `d_model` 768, `d_k = d_v = 64` |
| embedding | ALBERT-style factorised: `V×128` then `128→768` (38.6M → 6.5M) |
| MLM head | tied to the word-embedding table |
| sequence length | 128, contiguous packing (no padding, so no attention mask needed) |
| masking | BERT 80/10/10, re-randomised every batch |

No `[CLS]`: OLMo has none and NSP was dropped, so downstream classification **mean-pools**
the encoder hidden states (Sentence-BERT finds this often beats `[CLS]` anyway).

## Training history

| phase | data | tokens | result |
|---|---|---|---|
| 1 | WikiText-103 | 119M | val_loss 2.53, masked_acc 55.7% |
| 2 | UltraFineWeb | 681M (2 epochs) | val_loss 2.69, masked_acc 53.4% |
| 3 | UFW + TinyStories + IMDB (replay mix) | 425M | val_loss 2.09 |
| SST-2 | 67K labelled | — | **85.7% ± 1.2%** |

Phase val losses are **not comparable across rows** — each phase evaluates on its own
mixture, and TinyStories is far more predictable than web text. Only the SST-2 number is
measured the same way throughout.

## Things worth knowing

- **The model is capacity-bound, not data-bound.** Val loss plateaued at the eval-noise
  floor, zero dropout produced no overfitting, and it saw ~31 tokens per parameter. More
  parameters will help; more tokens will not.
- **The cosine schedule decays to zero**, so every continued-pretraining phase needed a
  re-warm. Phase 2 paid a +3.03 opening loss spike; phase 3 paid +0.23, because replay
  kept 39% of its mixture in-distribution. Use WSD (as `continuation.py` does) for any
  run that might be extended.
- **Model selection never touches the reported set.** The fine-tune carves a selection
  split out of train and evaluates the official 872-example dev set exactly once.
  Reporting the max over many evals on 872 examples is optimistic by 1–2 SE.
- **Budget by steps, not wall-clock** — a sleeping laptop must not spend its training
  budget while suspended.
- **The MLM is not a generator.** `notebooks/autoreg.ipynb` coerces it into one
  (3 trailing `[MASK]`s, commit slot 1, slide the window), then fixes the two obvious
  failures — never terminating, never copying from context — with ~10 minutes of CPU
  fine-tuning. Copy accuracy on held-out entities goes 0.8% → 93%.

See `WORKLOG.md` for the full history, including the bugs and what each one cost.
