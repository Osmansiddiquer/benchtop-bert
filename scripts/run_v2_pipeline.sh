#!/usr/bin/env bash
# v2 pipeline: 6 layers, d_ff 1792, GELU -- pretrain, checkpoint, anneal, evaluate.
#
#   Phase A  Cosmopedia 55% / UFW 35% / TinyStories 10%, WSD with NO decay
#            (--decay-frac 0) so it ends flat: the annealing is Phase B's job.
#            -> "$CDIR"/pre_anneal.pt, frozen and never rewritten.
#   Phase B  IMDB anneal, cosine -> 0, into a SEPARATE directory.
#   Phase C  SST-2 fine-tune + full benchmark (probe/finetune x sst2/stsb).
#
# Waits for the GPU to be free so it can be queued behind another run.
#
# Weights double as token budgets. Shifted toward Cosmopedia because it is the only
# genuinely NEW data for these weights: ckpt3 (which v2 inherits) already saw UFW for
# ~2.5 epochs across phases 2 and 3, so another full pass would push it near the ~4-epoch
# point where repeats stop paying. Cosmopedia it has never seen at all.
#   cosmopedia 1.00x of available (55%) | ufw 0.75x (35%) | tinystories 0.15x (10%)
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs checkpoints

STEPS_A=${STEPS_A:-22389}          # 22,389 * 32,768 = 733.7M tokens, exactly the
                                   # mix below -- more would push cosmopedia past 1.0x
STEPS_B=${STEPS_B:-1500}           # ~49M tokens of IMDB anneal
INIT=${INIT:-checkpoints/ckpt_v2_init.pt}
# Overridable so a re-run with different layer init writes to its own directories
# instead of clobbering the first run's checkpoints, metrics and results.
CDIR=${CDIR:-checkpoints/ckpt_v2}
IDIR=${IDIR:-checkpoints/ckpt_v2_imdb}
SDIR=${SDIR:-checkpoints/ckpt_sst2_v2arch}
TAG=${TAG:-v2arch}

# Wait for whatever else is training to finish.
while ps -eo args | awk '/(training\.jepa|training\.pretrain|finetune_sst2|evaluation\.benchmark)/ && !/awk/ && !/pretrain_v2/' | grep -q .; do
  echo "[v2] waiting for the GPU..."; sleep 120
done

if [ ! -f "$CDIR"/pre_anneal.pt ]; then
  echo "[A] v2 pretraining: 733.7M tokens, WSD flat, warmup 0.10"
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
    --init-from "$INIT" \
    --ckpt-dir "$CDIR" --ckpt-every 200 --eval-every 200 --log-every 20 \
    --device cuda --seed 0 >> logs/train_$TAG.log 2>&1
  cp "$CDIR"/last.pt "$CDIR"/pre_anneal.pt
  echo "[A] done -> "$CDIR"/pre_anneal.pt (frozen)"
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
    --device cuda --seed 0 >> logs/train_${TAG}_imdb.log 2>&1
  echo "[B] done -> "$IDIR"/last.pt"
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
