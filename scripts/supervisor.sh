#!/usr/bin/env bash
# Cron-driven supervisor for the phase-2 -> SST-2 pipeline.
#
# Why this exists: every watchdog we had lived inside a process that the laptop's
# sleep suspended along with the training itself, so nothing was left awake to
# notice the stall. Cron restarts with the system, so this script is the piece
# that survives a sleep/wake cycle and restarts the work within ~2 minutes.
#
# It is a pure state machine and takes no arguments. Safe to run at any moment:
# flock serialises ticks, and every branch is a no-op when the work is already
# running.
#
#   results exist            -> done, nothing to do
#   train/finetune running   -> nothing to do
#   ckpt step >= max_steps   -> pretraining finished, launch the SST-2 fine-tune
#   otherwise                -> (re)launch phase-2, which resumes from checkpoints/ckpt2/last.pt
#
# Also keeps the dashboard alive, since it dies to sleep the same way.
set -uo pipefail
cd "$(dirname "$0")" || exit 1

PY=/home/nimda/miniconda3/bin/python
LOGDIR=logs
mkdir -p "$LOGDIR"
SUPLOG="$LOGDIR/supervisor.log"
say() { echo "[$(date '+%F %T')] $*" >> "$SUPLOG"; }

# Only one tick at a time; a slow torch.load must not overlap the next cron fire.
#
# CRITICAL: every long-lived process spawned below MUST get `9>&-` to close this
# lock fd. Children inherit open fds, so a daemon that keeps fd 9 open holds the
# lock for its entire lifetime and every later cron tick dies here at `flock -n`.
# That happened for real: the dashboard captured the lock and silently disabled
# the supervisor for 5 h. The failure is invisible -- cron fires, the script exits
# 0, and nothing is logged.
exec 9>"$LOGDIR/.supervisor.lock"
flock -n 9 || exit 0

# Done only when the fine-tune says it finished all epochs. The results file is now
# rewritten after every epoch (so a sleep still leaves a usable number), so mere
# existence would wrongly latch this to "done" after epoch 1.
if [ -f results/finetune_sst2_results.json ]; then
  "$PY" -c "
import json,sys
try: sys.exit(0 if json.load(open('results/finetune_sst2_results.json')).get('complete') else 1)
except Exception: sys.exit(1)
" && exit 0
fi

# Bracket the dot rather than backslash-escaping it: awk warns on `\.` and the
# warnings drown the cron log.
running() { ps -eo comm,args | awk -v p="$1" '$1=="python" && $0 ~ p' | wc -l; }

# --- dashboard: restart if it is not up ---------------------------------------
if [ "$(running 'serve_dashboard[.]py')" -eq 0 ]; then
  setsid "$PY" tools/serve_dashboard.py --port 8000 --metrics checkpoints/ckpt2/metrics.jsonl \
    >> "$LOGDIR/dash.log" 2>&1 < /dev/null 9>&- &
  disown
  say "dashboard was down -> restarted"
fi

# --- training / fine-tune ------------------------------------------------------
[ "$(running 'train[.]py')" -gt 0 ] && exit 0
[ "$(running 'finetune_sst2[.]py')" -gt 0 ] && exit 0

# Nothing is running. Decide from the checkpoint itself (not metrics.jsonl, which
# can run ahead of the last saved step) whether pretraining actually finished.
read -r STEP MAXS <<< "$("$PY" - <<'EOF' 2>/dev/null
import torch
try:
    ck = torch.load('checkpoints/ckpt2/last.pt', map_location='cpu', weights_only=False)
    print(ck['step'], ck['cfg']['max_steps'])
except Exception:
    print(-1, -1)
EOF
)"
STEP=${STEP:--1}; MAXS=${MAXS:--1}

if [ "$STEP" -lt 0 ]; then
  say "could not read checkpoints/ckpt2/last.pt -> deferring to launch_phase2.sh"
  bash scripts/launch_phase2.sh >> "$SUPLOG" 2>&1 9>&-
  exit 0
fi

if [ "$STEP" -ge "$MAXS" ]; then
  # Cap relaunches: a fine-tune that crashes on startup (e.g. OOM) would otherwise be
  # respawned every 2 min forever, pinning the GPU for nothing.
  TRIES=$(cat "$LOGDIR/.ft_attempts" 2>/dev/null || echo 0)
  if [ "$TRIES" -ge 3 ]; then
    say "fine-tune failed $TRIES times -- giving up, inspect $LOGDIR/finetune.log"
    exit 0
  fi
  echo $((TRIES + 1)) > "$LOGDIR/.ft_attempts"
  say "phase-2 COMPLETE at step $STEP/$MAXS -> launching SST-2 fine-tune (attempt $((TRIES + 1)))"
  setsid "$PY" -m mini_enc_transformer.training.finetune_sst2 --device cuda --epochs 3 \
    >> "$LOGDIR/finetune.log" 2>&1 < /dev/null 9>&- &
  disown
else
  say "train.py not running at step $STEP/$MAXS -> resuming phase-2"
  bash scripts/launch_phase2.sh >> "$SUPLOG" 2>&1 9>&-
fi
