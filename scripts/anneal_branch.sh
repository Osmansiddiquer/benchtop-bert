#!/usr/bin/env bash
# Branched anneal: same checkpoint, same schedule, ONE corpus each, so the only
# difference between branches is data.
#
# Cosine to zero with NO warmup. Phase A ran WSD with --decay-frac 0, so it ended flat
# at 2e-4 with its decay phase never spent -- this IS that decay phase, which is why it
# starts at full LR rather than re-warming.
#
# MEASUREMENT WARNING. Comparing the branches' own val_loss is meaningless: each is
# annealed on a different distribution and TinyStories is intrinsically far easier, so
# it will report a lower loss whether or not it produced a better model. Both branches
# must be scored on (a) the SAME held-out mix, via tools/span_eval.py with an identical
# --data-name, and (b) the downstream benchmark. Only those are comparable.
#
#   CORPUS=tinystories TAG=tiny   bash scripts/anneal_branch.sh
#   CORPUS=cosmopedia  TAG=cosmo  bash scripts/anneal_branch.sh
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs checkpoints

CORPUS=${CORPUS:?set CORPUS, e.g. tinystories}
TAG=${TAG:?set TAG, e.g. tiny}
STEPS=${STEPS:-4000}
LR=${LR:-2e-4}
SRC=${SRC:-checkpoints/ckpt_v2/pre_anneal.pt}
CDIR=${CDIR:-checkpoints/ckpt_v2_anneal_$TAG}

# Wait for the GPU so branches queue rather than fight over 4GB. Checked via free VRAM,
# not process names: a name-based `ps | awk` pattern matches ANY command line containing
# the pattern string -- including an interactive shell that merely mentions it -- and
# wedges the loop forever on a phantom match.
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 800 ]; do
  echo "[anneal-$TAG] waiting for the GPU ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader) in use)..."
  sleep 60
done

echo "[anneal-$TAG] cosine $LR -> 0, no warmup, $STEPS steps, corpus=$CORPUS"
python -u -m mini_enc_transformer.training.pretrain_v2 \
  --data-dir data --data-name "$CORPUS" \
  --seq-len 128 --micro-batch 16 --grad-accum 16 \
  --n-layers 6 --d-ff 1792 \
  --lr "$LR" --weight-decay 0.01 \
  --schedule cosine --warmup-frac 0 \
  --max-steps "$STEPS" --grad-clip 1.0 --mlm-prob 0.15 \
  --mask-span-dist geometric --mask-geom-p 0.2 --mask-span-min 1 --mask-span-max 10 \
  --eval-span-min 1 --eval-span-max 1 \
  --probe-layers 4,5 \
  --init-from "$SRC" \
  --ckpt-dir "$CDIR" --ckpt-every 200 --eval-every 200 --log-every 20 \
  --device cuda --seed 0 >> logs/anneal_$TAG.log 2>&1
echo "[anneal-$TAG] done -> $CDIR/last.pt"
