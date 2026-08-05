"""Stream UltraFineWeb -> OLMo-tokenize -> pack into a local uint16 memmap.

This is the *network* stage, deliberately decoupled from training so that an
internet outage (up to and beyond 12h) can never interrupt a training run:
training reads only the finished local memmap. The stage is fully resumable and
tolerant of dropped connections --

  * Output is a pre-allocated `.bin` uint16 memmap of exactly --target-tokens.
  * A `.manifest.json` records tokens_written / docs_seen and is committed
    atomically every --flush-every docs. Completed bytes are never rewritten.
  * On restart (or after a network error) we reopen the memmap at tokens_written,
    rebuild the streaming iterator with the same seed, skip the docs already
    consumed, and continue. Network errors trigger exponential backoff (cap 60s)
    and unbounded retries, so a 12h outage just blocks-and-resumes.

Usage:
    python -m mini_enc_transformer.data.prep --target-tokens 1_500_000_000 --out data --min-score 0.9
    # kill any time (Ctrl-C / sleep); rerun the same command to resume.

uint16 is safe: OLMo vocab (50280) + the training-time [MASK] (50280) < 65536,
and [MASK] is never stored here (masking happens at train time).
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Resumable UltraFineWeb -> uint16 memmap packer")
    p.add_argument("--dataset", default="openbmb/Ultra-FineWeb")
    p.add_argument("--split", default="en")
    p.add_argument("--config", default=None, help="dataset config name (e.g. wikitext-103-raw-v1)")
    # Direct-shard mode: stream specific parquet shards instead of resolving the
    # whole (2048-shard, 9.7TB) repo -- full-dataset resolution is very slow.
    p.add_argument("--ufw-shards", type=int, default=8,
                   help="stream the first N UltraFineWeb en parquet shards directly (0 = use --dataset resolution)")
    p.add_argument("--local-parquet", default=None,
                   help="tokenize local .parquet files (dir or glob) instead of streaming from HF")
    p.add_argument("--tokenizer", default="allenai/OLMo-1B-hf")
    p.add_argument("--target-tokens", type=int, default=1_500_000_000)
    p.add_argument("--out", default="data")
    p.add_argument("--name", default="ultrafineweb_en")
    p.add_argument("--min-score", type=float, default=0.9)
    p.add_argument("--shuffle-buffer", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--flush-every", type=int, default=2_000, help="commit manifest every N docs")
    p.add_argument("--max-backoff", type=float, default=60.0)
    return p.parse_args()


def atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fchsync(f.fileno()) if hasattr(os, "fchsync") else os.fsync(f.fileno())
    os.replace(tmp, path)


def load_manifest(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


UFW_SHARD = ("https://huggingface.co/datasets/openbmb/Ultra-FineWeb/resolve/main/"
             "data/ultrafineweb_en/ultrafineweb-en-part-{:04d}-of-2048.parquet")


def build_stream(args, skip_docs):
    """Create the shuffled, score-filtered streaming iterator, skipping the first
    `skip_docs` documents so a resume lands exactly where it left off."""
    from datasets import load_dataset

    if args.local_parquet:
        # Tokenize locally-downloaded parquet files (fast; sidesteps the slow HF link).
        import glob as _glob
        pat = args.local_parquet
        files = sorted(_glob.glob(os.path.join(pat, "**", "*.parquet"), recursive=True)
                       if os.path.isdir(pat) else _glob.glob(pat))
        if not files:
            raise SystemExit(f"no .parquet files found under {pat}")
        print(f"[local] tokenizing {len(files)} parquet file(s) from {pat}", flush=True)
        ds = load_dataset("parquet", data_files=files, split="train", streaming=True)
    elif args.ufw_shards and args.dataset == "openbmb/Ultra-FineWeb":
        # Stream specific parquet shards by URL -- avoids resolving all 2048 shards.
        urls = [UFW_SHARD.format(i) for i in range(1, args.ufw_shards + 1)]
        ds = load_dataset("parquet", data_files=urls, split="train", streaming=True)
    elif args.config is not None:
        ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    else:
        ds = load_dataset(args.dataset, split=args.split, streaming=True)
    if args.shuffle_buffer > 0:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    it = iter(ds)
    # Deterministic skip: same seed+buffer => same document order across runs.
    for _ in range(skip_docs):
        next(it)
    return it


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    bin_path = os.path.join(args.out, args.name + ".bin")
    man_path = os.path.join(args.out, args.name + ".manifest.json")

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = tk.eos_token_id
    assert eos_id is not None and eos_id < 65536

    # Pre-allocate (or reopen) the fixed-size uint16 memmap.
    man = load_manifest(man_path)
    resuming = man is not None and os.path.exists(bin_path)
    if resuming:
        if man["target_tokens"] != args.target_tokens:
            sys.exit(f"target-tokens changed ({man['target_tokens']} -> {args.target_tokens}); "
                     f"delete {bin_path}/{man_path} to start fresh.")
        tokens_written = man["tokens_written"]
        docs_seen = man["docs_seen"]
        arr = np.memmap(bin_path, dtype=np.uint16, mode="r+", shape=(args.target_tokens,))
        print(f"[resume] {tokens_written:,}/{args.target_tokens:,} tokens, {docs_seen:,} docs seen", flush=True)
    else:
        arr = np.memmap(bin_path, dtype=np.uint16, mode="w+", shape=(args.target_tokens,))
        tokens_written, docs_seen = 0, 0
        # Record the real source: with --local-parquet the --dataset default is
        # meaningless and writing it produces a manifest that lies about its origin.
        src = f"local-parquet:{args.local_parquet}" if args.local_parquet else args.dataset
        man = {"dataset": src, "split": args.split, "tokenizer": args.tokenizer,
               "target_tokens": args.target_tokens, "min_score": args.min_score,
               "seed": args.seed, "shuffle_buffer": args.shuffle_buffer,
               "eos_id": eos_id, "dtype": "uint16", "tokens_written": 0, "docs_seen": 0}
        atomic_write_json(man_path, man)
        print(f"[fresh] allocating {args.target_tokens:,} uint16 "
              f"({args.target_tokens*2/1e9:.2f} GB) at {bin_path}", flush=True)

    def commit():
        arr.flush()
        man["tokens_written"] = tokens_written
        man["docs_seen"] = docs_seen
        atomic_write_json(man_path, man)

    dropped_short = 0
    backoff = 1.0
    t0 = time.time()
    since_flush = 0

    while tokens_written < args.target_tokens:
        try:
            it = build_stream(args, skip_docs=docs_seen)
            backoff = 1.0  # stream rebuilt successfully
            for ex in it:
                if tokens_written >= args.target_tokens:
                    break
                docs_seen += 1
                since_flush += 1
                score = ex.get("score", 1.0)  # datasets without a quality score keep everything
                try:
                    score = float(score)          # UltraFineWeb stores score as a string
                except (TypeError, ValueError):
                    score = 1.0
                if score < args.min_score:
                    continue
                text = ex.get("content") or ex.get("text") or ""
                if not text:
                    continue
                ids = tk(text, add_special_tokens=False)["input_ids"]
                if len(ids) < 8:            # skip near-empty docs
                    dropped_short += 1
                    continue
                ids.append(eos_id)          # document separator
                n = min(len(ids), args.target_tokens - tokens_written)
                arr[tokens_written:tokens_written + n] = np.asarray(ids[:n], dtype=np.uint16)
                tokens_written += n

                if since_flush >= args.flush_every:
                    commit()
                    since_flush = 0
                    rate = tokens_written / max(1e-9, time.time() - t0)
                    print(f"  {tokens_written:,}/{args.target_tokens:,} tokens "
                          f"({100*tokens_written/args.target_tokens:4.1f}%) | "
                          f"{docs_seen:,} docs | {rate/1e3:,.0f} tok/s | dropped_short={dropped_short:,}",
                          flush=True)
            else:
                # Iterator exhausted before hitting target (dataset ran out).
                if tokens_written < args.target_tokens:
                    print(f"[warn] stream exhausted at {tokens_written:,} tokens "
                          f"(< target {args.target_tokens:,}); stopping.", flush=True)
                break
        except (StopIteration,):
            break
        except Exception as e:  # network drops, transient HF errors -> backoff + resume
            commit()
            print(f"[retry] {type(e).__name__}: {e} -- sleeping {backoff:.0f}s then resuming "
                  f"from doc {docs_seen:,}", flush=True)
            time.sleep(backoff)
            backoff = min(args.max_backoff, backoff * 2)

    commit()
    print(f"[done] {tokens_written:,} tokens across {docs_seen:,} docs "
          f"in {time.time()-t0:.0f}s. memmap={bin_path}", flush=True)


if __name__ == "__main__":
    main()
