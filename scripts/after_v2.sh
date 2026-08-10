#!/usr/bin/env bash
# After the v2 pipeline finishes: judge whether the stacked layers ever became their
# own layers, and if they did not, rebuild with RANDOM init and run the whole thing
# again into separate directories.
#
# The judgment is not a guess. L4 was a copy of L2 and L5 a copy of L3; naturally
# trained adjacent layers sit at weight cosine ~0.00 on this model. A child still above
# cos 0.5, or with >50% of its neurons never driven positive, has not become a layer --
# it is either a duplicate or one the model has switched off.
#
# The first run's checkpoints, metrics and results are never touched: the re-run writes
# to ckpt_v2r / ckpt_v2r_imdb / ckpt_sst2_v2r and results/v2r-*.
set -uo pipefail
cd "$(dirname "$0")/.."

FIRST_RESULT=results/v2arch-benchmarks-evals.json

echo "[after-v2] waiting for the first pipeline to finish..."
while [ ! -f "$FIRST_RESULT" ]; do
  if ! ps -eo args | awk '/(run_v2_pipeline|pretrain_v2|finetune_sst2|evaluation\.benchmark)/ && !/awk/' | grep -q .; then
    echo "[after-v2] pipeline stopped without producing $FIRST_RESULT -- not re-running"
    exit 1
  fi
  sleep 120
done
echo "[after-v2] first pipeline complete"

CK=checkpoints/ckpt_v2/pre_anneal.pt
[ -f "$CK" ] || CK=checkpoints/ckpt_v2/last.pt

# exit status, read directly rather than through a pipe
if python -u tools/judge_stacked_layers.py --ckpt "$CK" > logs/judge_v2.log 2>&1; then
  echo "[after-v2] stacked layers DIFFERENTIATED -- keeping the run, nothing to do"
  cat logs/judge_v2.log
  exit 0
fi

echo "[after-v2] stacked layers did NOT differentiate:"
grep -vE "Warning|scipy" logs/judge_v2.log | tail -6

echo "[after-v2] rebuilding init with RANDOM layers 4-5"
python -u tools/build_v2_from_ckpt3.py --new-layer-init random \
  --out checkpoints/ckpt_v2r_init.pt >> logs/build_v2r.log 2>&1 || {
    echo "[after-v2] rebuild failed, see logs/build_v2r.log"; exit 1; }

echo "[after-v2] re-running the identical pipeline with the new init"
INIT=checkpoints/ckpt_v2r_init.pt \
CDIR=checkpoints/ckpt_v2r IDIR=checkpoints/ckpt_v2r_imdb \
SDIR=checkpoints/ckpt_sst2_v2r TAG=v2r \
  bash scripts/run_v2_pipeline.sh >> logs/v2r_pipeline.log 2>&1
echo "[after-v2] re-run complete -> results/v2r-benchmarks-evals.json"
