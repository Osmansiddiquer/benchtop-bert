#!/usr/bin/env bash
# Fires when JEPA phase A reaches max_steps: IMDB anneal, then SST-2.
#
# The anneal uses the JEPA objective, not MLM. Measured reason: after ~5,000 JEPA
# steps the encoder had drifted 27.7% from ckpt3 while mlm_head.dense and .norm were
# *exactly* unchanged (0.00000) -- the MLM head is a decoder for a representation that
# no longer exists. An MLM anneal would spend its short budget re-fitting that head
# instead of adapting to the review domain, and SST-2 uses the encoder only.
#
# Waits for the checkpoint to REACH max_steps, not merely for the process to vanish,
# so a crash is not mistaken for completion.
set -uo pipefail
cd "$(dirname "$0")/.."

STEPS_B=${STEPS_B:-1500}

while true; do
  if ! ps -eo args | awk '/training\.jepa/ && !/awk/' | grep -q .; then
    read -r S M <<< "$(python - <<'PY' 2>/dev/null
import torch
try:
    c = torch.load('checkpoints/ckpt_jepa/last.pt', map_location='cpu', weights_only=False)
    print(c['step'], c['cfg']['max_steps'])
except Exception:
    print(-1, -1)
PY
)"
    if [ "${S:--1}" -ge "${M:-999999}" ] && [ "${S:--1}" -gt 0 ]; then
      echo "[chain] phase A complete at $S/$M"
      break
    fi
    echo "[chain] training gone but checkpoint at ${S}/${M} -- stopping, needs a look"
    exit 1
  fi
  sleep 60
done

# Freeze the pre-anneal model under its own name before anything else runs.
cp checkpoints/ckpt_jepa/last.pt checkpoints/ckpt_jepa/pre_anneal.pt
echo "[chain] preserved checkpoints/ckpt_jepa/pre_anneal.pt"

echo "[chain] IMDB anneal (JEPA objective, cosine -> 0)"
python -u -m mini_enc_transformer.training.jepa \
  --data-dir data --data-name "imdb:0.70,cosmopedia:0.15,ultrafineweb_en:0.15" \
  --seq-len 128 --micro-batch 16 --grad-accum 16 \
  --lr 5e-5 --schedule cosine --warmup-frac 0.02 \
  --max-steps "$STEPS_B" --mlm-prob 0.15 \
  --mask-span-dist geometric --mask-geom-p 0.2 --mask-span-min 1 --mask-span-max 10 \
  --top-k-layers 3 --ema-start 0.996 --ema-end 0.999 \
  --init-from checkpoints/ckpt_jepa/pre_anneal.pt \
  --ckpt-dir checkpoints/ckpt_jepa_imdb --ckpt-every 200 --eval-every 100 --log-every 20 \
  --device cuda --seed 0 >> logs/train_jepa_imdb.log 2>&1
echo "[chain] anneal done"

echo "[chain] SST-2 fine-tune of the post-anneal model"
python -u -m mini_enc_transformer.training.finetune_sst2 \
  --ckpt checkpoints/ckpt_jepa_imdb/last.pt \
  --run-dir checkpoints/ckpt_sst2_jepa \
  --out results/finetune_sst2_results_jepa.json \
  --device cuda --epochs 3 >> logs/finetune_jepa.log 2>&1

echo "[chain] full benchmark on the post-anneal model"
python -u -m mini_enc_transformer.evaluation.benchmark \
  --ckpt checkpoints/ckpt_jepa_imdb/last.pt --tasks sst2,stsb --modes probe,finetune \
  --out results/jepa-benchmarks-evals.json --device cuda >> logs/benchmark_jepa.log 2>&1
echo "[chain] done"
