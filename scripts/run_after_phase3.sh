#!/usr/bin/env bash
# Wait for phase 3 to reach max_steps, then fine-tune ckpt3 on SST-2 with the new
# defaults (llrd 1.0, lr 3e-5). Explicit --ckpt: the script would otherwise prefer
# checkpoints/ckpt_sst2/best.pt (the v2 fine-tune) over the new backbone.
cd /home/nimda/repos/transformer/transformer-v1
while true; do
  if [ "$(ps -eo comm,args | awk '$1=="python" && /train[.]py/' | wc -l)" -eq 0 ]; then
    read -r S M <<< "$(python - <<'PY'
import torch
try:
    c=torch.load('checkpoints/ckpt3/last.pt',map_location='cpu',weights_only=False)
    print(c['step'], c['cfg']['max_steps'])
except Exception: print(-1,-1)
PY
)"
    if [ "${S:--1}" -ge "${M:-0}" ] && [ "${S:--1}" -gt 0 ]; then
      echo "phase 3 complete at $S/$M -> launching SST-2 fine-tune"
      exec python -m mini_enc_transformer.training.finetune_sst2 --ckpt checkpoints/ckpt3/last.pt --run-dir ckpt_sst2_v3 \
        --out results/finetune_sst2_results_v3.json --device cuda --epochs 3
    fi
    echo "train.py gone but ckpt3 at $S/$M (incomplete) -- stopping, needs a look"; exit 1
  fi
  sleep 60
done
