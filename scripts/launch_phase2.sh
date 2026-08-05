#!/usr/bin/env bash
# Phase 2: continued pretraining of the laptop encoder on UltraFineWeb, warm-started
# from phase-1's checkpoint (checkpoints/ckpt/last.pt). Writes to checkpoints/ckpt2/ (separate metrics so the
# dashboard curves don't collide with phase-1's step numbering). Auto-sizes the cosine
# schedule to ~MIN(2 epochs of available data, STEP_CAP). Idempotent-ish: if ckpt2 is
# mid-run it resumes; otherwise it warm-starts fresh.
set -euo pipefail
cd "$(dirname "$0")"

MIN_TOKENS=${MIN_TOKENS:-80000000}   # don't start phase 2 until this many UFW tokens are packed
EPOCHS=${EPOCHS:-2.0}
STEP_CAP=${STEP_CAP:-30000}          # high enough not to clip a 2-epoch run over ~340M tokens
TOK_PER_STEP=32768                    # micro 16 * accum 16 * seq 128
# Durable log dir: this script also runs from cron, where CLAUDE_JOB_DIR is unset
# and the old path did not exist -- the failed redirect silently stopped training.
mkdir -p logs
LOG=logs/train_phase2.log

if [ ! -f checkpoints/ckpt/last.pt ]; then echo "phase-1 checkpoint missing; not starting"; exit 3; fi

TW=$(python -c "import json;print(json.load(open('data/ultrafineweb_en.manifest.json'))['tokens_written'])" 2>/dev/null || echo 0)

# Already running? (detect REAL python processes; pgrep -f self-matches shells)
if [ "$(ps -eo comm,args | awk '$1=="python" && /train\.py/' | wc -l)" -gt 0 ]; then
  echo "a train.py is already running; nothing to do"; exit 0; fi

if [ -f checkpoints/ckpt2/last.pt ]; then
  # Resume existing weights/logs; recompute the budget so a raised EPOCHS/cap takes effect
  # (the checkpoint's own cfg may carry an older, smaller max_steps).
  MAXSTEPS=$(python -c "print(min($STEP_CAP, int($EPOCHS*$TW/$TOK_PER_STEP)))")
  echo "resuming phase-2 from checkpoints/ckpt2/last.pt -> max_steps $MAXSTEPS ($(python -c "print(round($MAXSTEPS*$TOK_PER_STEP/$TW,2))") epochs)"
  RESUME="--resume"; INIT=""
else
  if [ "$TW" -lt "$MIN_TOKENS" ]; then
    echo "UltraFineWeb has only $TW tokens (< $MIN_TOKENS); waiting for more before phase 2"; exit 4
  fi
  MAXSTEPS=$(python -c "print(min($STEP_CAP, int($EPOCHS*$TW/$TOK_PER_STEP)))")
  ACTUAL_EP=$(python -c "print(round($MAXSTEPS*$TOK_PER_STEP/$TW, 2))")
  echo "starting phase-2: init from checkpoints/ckpt/last.pt, $TW UFW tokens -> $MAXSTEPS steps (${ACTUAL_EP} epochs actual)"
  RESUME=""; INIT="--init-from checkpoints/ckpt/last.pt"
fi

setsid python -m mini_enc_transformer.training.pretrain \
  --data-dir data --data-name ultrafineweb_en \
  --d-model 768 --d-embed 128 --n-heads 4 --n-layers 4 --d-k 64 --d-v 64 --seq-len 128 \
  --micro-batch 16 --grad-accum 16 --lr 5e-4 --weight-decay 0.01 --warmup-frac 0.04 \
  --max-steps "$MAXSTEPS" --grad-clip 1.0 --mlm-prob 0.15 \
  --ckpt-dir ckpt2 --ckpt-every 200 --eval-every 400 --log-every 20 --device cuda --seed 0 \
  $RESUME $INIT >> "$LOG" 2>&1 < /dev/null &
disown
echo "phase-2 launched (log: $LOG)"

# Point the dashboard at phase-2 metrics (restart it on the same port).
pgrep -f 'serve_dashboard.py' | xargs -r kill 2>/dev/null || true
sleep 1
setsid python tools/serve_dashboard.py --port 8000 --metrics checkpoints/ckpt2/metrics.jsonl \
  >> logs/dash.log 2>&1 < /dev/null &
disown
echo "dashboard repointed to checkpoints/ckpt2/metrics.jsonl on http://localhost:8000"
