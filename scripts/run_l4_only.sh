#!/usr/bin/env bash
# Train layer 4 and NOTHING else, from step 0, on the same WSD schedule.
#
# Why: L4 was measured worth -0.007 nats after two different initialisations, and the
# pairwise ablation showed nothing was covering for it. The diagnosis was that a
# jointly-trained stack simply routes around a layer that arrives late -- L1/L2/L5 had
# already partitioned the available work between them.
#
# Freezing removes the escape route. With every other parameter fixed -- embeddings,
# MLM head, and blocks 0,1,2,3,5 -- the loss can only improve through L4. If L4 still
# converges to zero contribution under those conditions, the position genuinely has no
# work available and the answer is architectural, not optimisational.
#
# L4 is reset to standard init, N(0, 0.02), by tools/restack_layers.py --order 0,1,2,3,NEW,5.
# Everything else about the run is unchanged from the main v2 config: same data mix,
# same lr 2e-4, same weight decay, same WSD (warmup 0.10, no decay), same 22,389 steps.
# No LR boost, no re-warm, no special treatment.
#
# Writes to its OWN directory: the step counter restarts at 0, which would corrupt the
# main run's metrics timeline (14,200 -> 0). checkpoints/ckpt_v2/last.pt is untouched and
# still resumable via scripts/run_v2_graft.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs checkpoints

STEPS=${STEPS:-22389}
INIT=${INIT:-checkpoints/ckpt_v2_l4only_init.pt}
CDIR=${CDIR:-checkpoints/ckpt_v2_l4only}
LOG=${LOG:-logs/train_l4only.log}

RESUME=""
[ -f "$CDIR"/last.pt ] && RESUME="--resume"

echo "[l4only] training layer 4 only, frozen stack, from step 0"
python -u -m mini_enc_transformer.training.pretrain_v2 \
  --data-dir data \
  --data-name "cosmopedia:403451249:403451249,ultrafineweb_en:256832885:256832885,tinystories:73377468:73377468" \
  --seq-len 128 --micro-batch 16 --grad-accum 16 \
  --n-layers 6 --d-ff 1792 \
  --lr 2e-4 --weight-decay 0.01 \
  --schedule wsd --warmup-frac 0.10 --decay-frac 0 \
  --max-steps "$STEPS" --grad-clip 1.0 --mlm-prob 0.15 \
  --mask-span-dist geometric --mask-geom-p 0.2 --mask-span-min 1 --mask-span-max 10 \
  --eval-span-min 1 --eval-span-max 1 \
  --freeze-except-layers 4 --probe-layers 4,5 \
  --init-from "$INIT" $RESUME \
  --ckpt-dir "$CDIR" --ckpt-every 200 --eval-every 200 --log-every 20 \
  --device cuda --seed 0 >> "$LOG" 2>&1
echo "[l4only] done -> $CDIR/last.pt"
