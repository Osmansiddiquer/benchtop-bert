#!/usr/bin/env bash
# JEPA pipeline: latent-target pretraining from ckpt3, then an IMDB anneal, then SST-2.
#
#   Phase A  500M tokens, Cosmopedia 70% / UltraFineWeb 30%, WSD with NO decay,
#            SpanBERT geometric span lengths (p=0.2, clip 10, mean ~3.8)
#            (decay-frac 0) so it ends flat -- the annealing is Phase B's job.
#            -> checkpoints/ckpt_jepa/last.pt, copied to pre_anneal.pt and never
#               touched again, so the pre-IMDB model is preserved.
#   Phase B  IMDB anneal, cosine -> 0, writing to a SEPARATE directory.
#   Phase C  SST-2 fine-tune of the post-anneal model only.
#
# Baseline to beat: ckpt3 = 85.67% +/- 1.19% on SST-2 dev.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs checkpoints

TOK_PER_STEP=32768                 # micro 16 * accum 16 * seq 128
STEPS_A=${STEPS_A:-15258}          # 500M tokens
STEPS_B=${STEPS_B:-1500}           # ~49M tokens of IMDB anneal

# ---- Phase A -----------------------------------------------------------------
if [ ! -f checkpoints/ckpt_jepa/pre_anneal.pt ]; then
  echo "[A] JEPA 500M tokens, WSD flat, from ckpt3"
  python -m mini_enc_transformer.training.jepa \
    --data-dir data \
    --data-name "cosmopedia:350000000:350000000,ultrafineweb_en:150000000:150000000" \
    --seq-len 128 --micro-batch 16 --grad-accum 16 \
    --lr 1e-4 --weight-decay 0.01 \
    --schedule wsd --warmup-frac 0.05 --decay-frac 0 \
    --max-steps "$STEPS_A" --mlm-prob 0.15 --mask-span-dist geometric --mask-geom-p 0.2 \
    --mask-span-min 1 --mask-span-max 10 \
    --top-k-layers 3 --ema-start 0.999 --ema-end 0.9999 \
    --init-from checkpoints/ckpt3/last.pt \
    --ckpt-dir checkpoints/ckpt_jepa --ckpt-every 200 --eval-every 200 --log-every 20 \
    --device cuda --seed 0 >> logs/train_jepa.log 2>&1
  # Freeze the pre-anneal model under its own name. Phase B writes elsewhere, but an
  # explicit copy means no later run can clobber it by reusing --ckpt-dir.
  cp checkpoints/ckpt_jepa/last.pt checkpoints/ckpt_jepa/pre_anneal.pt
  echo "[A] done -> checkpoints/ckpt_jepa/pre_anneal.pt (preserved)"
fi

# ---- Phase B -----------------------------------------------------------------
if [ ! -f checkpoints/ckpt_jepa_imdb/last.pt ]; then
  echo "[B] IMDB anneal, cosine -> 0"
  python -m mini_enc_transformer.training.jepa \
    --data-dir data \
    --data-name "imdb:0.70,cosmopedia:0.15,ultrafineweb_en:0.15" \
    --seq-len 128 --micro-batch 16 --grad-accum 16 \
    --lr 5e-5 --schedule cosine --warmup-frac 0.02 \
    --max-steps "$STEPS_B" --mlm-prob 0.15 --mask-span-dist geometric --mask-geom-p 0.2 \
    --mask-span-min 1 --mask-span-max 10 \
    --top-k-layers 3 --ema-start 0.999 --ema-end 0.9999 \
    --init-from checkpoints/ckpt_jepa/pre_anneal.pt \
    --ckpt-dir checkpoints/ckpt_jepa_imdb --ckpt-every 200 --eval-every 100 --log-every 20 \
    --device cuda --seed 0 >> logs/train_jepa_imdb.log 2>&1
  echo "[B] done -> checkpoints/ckpt_jepa_imdb/last.pt"
fi

# ---- Phase C -----------------------------------------------------------------
echo "[C] SST-2 fine-tune of the post-anneal model"
python -m mini_enc_transformer.training.finetune_sst2 \
  --ckpt checkpoints/ckpt_jepa_imdb/last.pt \
  --run-dir checkpoints/ckpt_sst2_jepa \
  --out results/finetune_sst2_results_jepa.json \
  --device cuda --epochs 3 >> logs/finetune_jepa.log 2>&1
echo "[C] done -> results/finetune_sst2_results_jepa.json"
