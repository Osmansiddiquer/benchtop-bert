#!/usr/bin/env bash
# Tokenize user-downloaded UltraFineWeb parquet shards (data/ufw_raw/*.parquet) into
# the local memmap, offline. Waits until ALL present shards are complete (pyarrow can
# read the footer) so an in-progress download can't corrupt the run. Idempotent:
# re-tokenizes if the number of complete shards grew since last time.
set -euo pipefail
cd "$(dirname "$0")"
RAW=data/ufw_raw
LOG="${CLAUDE_JOB_DIR:-/tmp}/tmp/prep_ufw.log"
MARK="$RAW/.ingested_shards"

mapfile -t FILES < <(find "$RAW" -name '*.parquet' 2>/dev/null | sort)
TOTAL=${#FILES[@]}
if [ "$TOTAL" -eq 0 ]; then echo "no parquet files in $RAW yet — waiting for the download"; exit 4; fi

# Count how many are fully written (pyarrow reads the footer at end-of-file).
VALID=$(python -c "
import sys, pyarrow.parquet as pq
ok=0
for f in sys.argv[1:]:
    try: pq.ParquetFile(f).metadata; ok+=1
    except Exception: pass
print(ok)
" "${FILES[@]}" 2>/dev/null || echo 0)

if [ "$VALID" -ne "$TOTAL" ]; then
  echo "waiting: $VALID/$TOTAL shards complete (a download is still in progress)"; exit 4
fi

if [ "$(ps -eo comm,args | awk '$1=="python" && /data_prep\.py/' | wc -l)" -gt 0 ]; then
  echo "tokenization already running"; exit 0; fi

TW=$(python -c "import json;print(json.load(open('data/ultrafineweb_en.manifest.json'))['tokens_written'])" 2>/dev/null || echo 0)
PREV=$(cat "$MARK" 2>/dev/null || echo 0)
if [ "$TW" -gt 1000000 ] && [ "$PREV" -ge "$TOTAL" ]; then
  echo "UFW already tokenized ($TW tokens from $PREV shard(s))"; exit 0
fi

echo "$TOTAL" > "$MARK"
rm -f data/ultrafineweb_en.bin data/ultrafineweb_en.manifest.json
setsid python -m mini_enc_transformer.data.prep --local-parquet "$RAW" --min-score 0.9 --shuffle-buffer 0 \
  --target-tokens 700000000 --out data --name ultrafineweb_en --flush-every 20000 \
  >> "$LOG" 2>&1 < /dev/null &
disown
echo "started local UFW tokenization from $TOTAL complete shard(s)"
