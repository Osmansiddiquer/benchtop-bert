#!/usr/bin/env bash
# Resume the v2 run after grafting layer 4, then carry on through the same Phase B/C
# as run_v2_pipeline.sh. Everything stays IN PLACE -- same checkpoint dir, same metrics
# file, same results names -- so the dashboard and every downstream script are unchanged
# and the step axis is continuous across the graft.
#
# What changed at the graft point (see tools/graft_l4.py):
#   * L4's W_q/W_k/W_v and l1 re-initialised; W_O and l2 ZEROED, so the layer emits
#     nothing at step 0 and cannot disturb the converged stack while it learns.
#     Measured loss delta across the graft: +0.0077 nats.
#   * L4's stale Adam moments dropped (--reset-opt-layers 4) -- they describe weights
#     that no longer exist, and Adam would otherwise divide fresh gradients by an old
#     second moment for hundreds of steps.
#   * Layers 4-5 get their own param group: 5x LR, ZERO weight decay. The zeroed output
#     projections receive only weak gradient until they lift off zero, and decay would
#     compete with exactly the signal meant to grow them.
#   * 400-step linear re-warm on top of the base schedule.
#
# NOTE ON L5: it is in the boosted group per the plan, but unlike L4 it was NOT
# re-initialised -- it is mid-recovery (never_pos 67.5%->59%, util 0.148->0.21, centred
# head_sim 0.42 against a -0.20 null). Raising its LR 5x is the one part of this with
# real downside risk; the l5_np / l5_ent traces on the dashboard show it immediately.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs checkpoints

STEPS_A=${STEPS_A:-22389}
STEPS_B=${STEPS_B:-1500}
CDIR=${CDIR:-checkpoints/ckpt_v2}
IDIR=${IDIR:-checkpoints/ckpt_v2_imdb}
SDIR=${SDIR:-checkpoints/ckpt_sst2_v2arch}
TAG=${TAG:-v2arch}
# Phase A/B keep the ORIGINAL log names so the whole run stays in one file; the
# results names keep TAG so they match what the monitor and after_v2 look for.
LOG_A=${LOG_A:-logs/train_v2.log}
LOG_B=${LOG_B:-logs/train_v2_imdb.log}
BOOST_LAYERS=${BOOST_LAYERS-4}   # L5 deliberately excluded: it was never
                                  # re-initialised and was recovering on its
                                  # own; 5x LR drove its never_pos back up
                                  # (64.6%% -> 69.2%%) over 1000 steps.
BOOST_MULT=${BOOST_MULT:-5.0}
REWARM=${REWARM:-400}

if [ ! -f "$CDIR"/pre_anneal.pt ]; then
  echo "[A] resuming v2 pretraining after the L4 graft"
  python -u -m mini_enc_transformer.training.pretrain_v2 \
    --data-dir data \
    --data-name "cosmopedia:403451249:403451249,ultrafineweb_en:256832885:256832885,tinystories:73377468:73377468" \
    --seq-len 128 --micro-batch 16 --grad-accum 16 \
    --n-layers 6 --d-ff 1792 \
    --lr 2e-4 --weight-decay 0.01 \
    --schedule wsd --warmup-frac 0.10 --decay-frac 0 \
    --max-steps "$STEPS_A" --grad-clip 1.0 --mlm-prob 0.15 \
    --mask-span-dist geometric --mask-geom-p 0.2 --mask-span-min 1 --mask-span-max 10 \
    --eval-span-min 1 --eval-span-max 1 \
    --resume --ckpt-dir "$CDIR" --ckpt-every 200 --eval-every 200 --log-every 20 \
    --boost-layers "$BOOST_LAYERS" --boost-mult "$BOOST_MULT" --boost-hold-frac 0.15 --boost-wd 0.0 \
    --rewarm-steps "$REWARM" --reset-opt-layers 4 --probe-layers 4,5 \
    --device cuda --seed 0 >> "$LOG_A" 2>&1
  cp "$CDIR"/last.pt "$CDIR"/pre_anneal.pt
  echo "[A] done -> $CDIR/pre_anneal.pt (frozen)"
fi

if [ ! -f "$IDIR"/last.pt ]; then
  echo "[B] IMDB anneal, cosine -> 0"
  python -u -m mini_enc_transformer.training.pretrain_v2 \
    --data-dir data --data-name "imdb:0.70,ultrafineweb_en:0.20,cosmopedia:0.10" \
    --seq-len 128 --micro-batch 16 --grad-accum 16 \
    --n-layers 6 --d-ff 1792 \
    --lr 5e-5 --schedule cosine --warmup-frac 0.02 \
    --max-steps "$STEPS_B" --mlm-prob 0.15 \
    --mask-span-dist geometric --mask-geom-p 0.2 --mask-span-min 1 --mask-span-max 10 \
    --eval-span-min 1 --eval-span-max 1 \
    --init-from "$CDIR"/pre_anneal.pt \
    --ckpt-dir "$IDIR" --ckpt-every 200 --eval-every 100 --log-every 20 \
    --probe-layers 4,5 \
    --device cuda --seed 0 >> "$LOG_B" 2>&1
  echo "[B] done -> $IDIR/last.pt"
fi

echo "[C] SST-2 fine-tune"
python -u -m mini_enc_transformer.training.finetune_sst2 \
  --ckpt "$IDIR"/last.pt --run-dir "$SDIR" \
  --out results/finetune_sst2_results_$TAG.json \
  --device cuda --epochs 3 >> logs/finetune_$TAG.log 2>&1

echo "[C] full benchmark"
python -u -m mini_enc_transformer.evaluation.benchmark \
  --ckpt "$IDIR"/last.pt --tasks sst2,stsb --modes probe,finetune \
  --out results/$TAG-benchmarks-evals.json --device cuda >> logs/benchmark_$TAG.log 2>&1
echo "[C] done"
