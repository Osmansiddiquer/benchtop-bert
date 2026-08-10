#!/usr/bin/env bash
# Downstream chain for the cosmopedia-annealed v2 checkpoint:
#   B  IMDB anneal, cosine -> 0
#   C  SST-2 fine-tune
#   D  full benchmark: probe/finetune x SST-2/STS-B
#
# Source is checkpoints/ckpt_v2_anneal_cosmo/last.pt (step 4000), itself the product of
# Phase A (18,400 steps) + a 4,000-step cosmopedia anneal. That checkpoint measured
# -0.054 scat_loss / +1.18pp span_acc against pre_anneal on the Phase-A mix, but +0.057
# nats WORSE on IMDB -- the two disagreed, which is why the downstream numbers are the
# thing that actually settles it.
#
# Writes to its own directories; nothing existing is overwritten.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs checkpoints results

SRC=${SRC:-checkpoints/ckpt_v2_anneal_cosmo/last.pt}
IDIR=${IDIR:-checkpoints/ckpt_v2_cosmo_imdb}
SDIR=${SDIR:-checkpoints/ckpt_sst2_v2cosmo}
TAG=${TAG:-v2cosmo}
STEPS_B=${STEPS_B:-1500}

# VRAM check, not a process-name grep: a name pattern matches any shell that merely
# mentions it, which wedged this loop once already.
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 800 ]; do
  echo "[$TAG] waiting for the GPU ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader) in use)..."
  sleep 60
done

if [ ! -f "$IDIR"/last.pt ]; then
  echo "[B] IMDB anneal, cosine -> 0, $STEPS_B steps   ($(date +%H:%M))"
  python -u -m mini_enc_transformer.training.pretrain_v2 \
    --data-dir data --data-name "imdb:0.70,ultrafineweb_en:0.20,cosmopedia:0.10" \
    --seq-len 128 --micro-batch 16 --grad-accum 16 \
    --n-layers 6 --d-ff 1792 \
    --lr 5e-5 --schedule cosine --warmup-frac 0.02 \
    --max-steps "$STEPS_B" --mlm-prob 0.15 \
    --mask-span-dist geometric --mask-geom-p 0.2 --mask-span-min 1 --mask-span-max 10 \
    --eval-span-min 1 --eval-span-max 1 \
    --probe-layers 4,5 \
    --init-from "$SRC" \
    --ckpt-dir "$IDIR" --ckpt-every 200 --eval-every 100 --log-every 20 \
    --device cuda --seed 0 >> logs/train_${TAG}_imdb.log 2>&1
  echo "[B] done -> $IDIR/last.pt   ($(date +%H:%M))"
fi

echo "[C] SST-2 fine-tune   ($(date +%H:%M))"
python -u -m mini_enc_transformer.training.finetune_sst2 \
  --ckpt "$IDIR"/last.pt --run-dir "$SDIR" \
  --out results/finetune_sst2_results_$TAG.json \
  --device cuda --epochs 3 >> logs/finetune_$TAG.log 2>&1
echo "[C] done   ($(date +%H:%M))"

echo "[D] full benchmark: probe/finetune x SST-2/STS-B   ($(date +%H:%M))"
python -u -m mini_enc_transformer.evaluation.benchmark \
  --ckpt "$IDIR"/last.pt --tasks sst2,stsb --modes probe,finetune \
  --out results/$TAG-benchmarks-evals.json --device cuda >> logs/benchmark_$TAG.log 2>&1
echo "[D] done -> results/$TAG-benchmarks-evals.json   ($(date +%H:%M))"
