#!/usr/bin/env bash
# Phase 4 (optional): domain anneal on movie reviews.
#
# Annealing-phase data curation -- data seen during the final low-LR decay has
# outsized influence on the finished model (MiniCPM WSD, Llama 3, OLMo 2's Dolmino
# mix), and DAPT (Gururangan et al. 2020) says in-domain unlabelled text lifts
# in-domain downstream tasks. SST-2 is movie reviews, so IMDB is the target domain.
#
# NOT part of the main pipeline: run this only AFTER the phase-3 SST-2 number is in,
# so it stays a single-variable change against a measured baseline.
#
#   bash anneal_imdb.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs ckpt4

STEPS=${STEPS:-1200}          # 1200 * 32768 = 39.3M tokens; IMDB share ~27.5M = 1.2 epochs
LR=${LR:-5e-5}                # phase 3 ends at LR~0, so this is a small bump, not a re-warm

[ -f checkpoints/ckpt3/last.pt ] || { echo "checkpoints/ckpt3/last.pt missing -- run phase 3 first"; exit 3; }

echo "annealing from checkpoints/ckpt3/last.pt: $STEPS steps, lr $LR, 70% IMDB / 30% replay"
setsid python -m mini_enc_transformer.training.pretrain \
  --data-dir data \
  --data-name "imdb:0.70,ultrafineweb_en:0.15,tinystories:0.15" \
  --d-model 768 --d-embed 128 --n-heads 4 --n-layers 4 --d-k 64 --d-v 64 --seq-len 128 \
  --micro-batch 16 --grad-accum 16 --lr "$LR" --weight-decay 0.01 --warmup-frac 0.02 \
  --max-steps "$STEPS" --grad-clip 1.0 --mlm-prob 0.15 \
  --ckpt-dir ckpt4 --ckpt-every 200 --eval-every 200 --log-every 20 --device cuda --seed 0 \
  --init-from checkpoints/ckpt3/last.pt >> logs/train_phase4.log 2>&1 < /dev/null &
disown
echo "phase 4 launched (log: logs/train_phase4.log)"
echo "when done, fine-tune with:"
echo "  python -m mini_enc_transformer.training.finetune_sst2 --ckpt checkpoints/ckpt4/last.pt --run-dir ckpt_sst2_v4 \\"
echo "    --out results/finetune_sst2_results_v4.json --device cuda --epochs 3"
