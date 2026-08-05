"""Memmap-backed dataset for MLM training.

Reads the uint16 token stream produced by `data_prep.py` and serves fixed-length
contiguous blocks. Because docs were packed end-to-end (separated by eos), every
block is full -- no padding, so no attention mask is needed (which suits the
current MHA, that only supports a causal flag, not a key-padding mask).

Masking is applied on the fly per batch by the training loop via
`mlm.mask_tokens`, so the same cached tokens get a fresh mask each epoch.
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedMemmapDataset(Dataset):
    def __init__(self, out_dir: str, name: str, seq_len: int, split: str = "train",
                 val_fraction: float = 0.01, limit_tokens: int = None):
        man_path = os.path.join(out_dir, name + ".manifest.json")
        bin_path = os.path.join(out_dir, name + ".bin")
        with open(man_path) as f:
            man = json.load(f)
        # Only read the tokens actually written (the memmap may be pre-allocated larger).
        n = man["tokens_written"] if limit_tokens is None else min(limit_tokens, man["tokens_written"])
        full = np.memmap(bin_path, dtype=np.uint16, mode="r", shape=(man["target_tokens"],))

        n_blocks = n // seq_len
        if n_blocks < 2:
            raise ValueError(f"only {n} tokens -> {n_blocks} blocks of {seq_len}; need more data")
        # Contiguous train/val split by block, val taken from the tail.
        n_val = max(1, int(n_blocks * val_fraction))
        n_train = n_blocks - n_val
        if split == "train":
            self.lo, self.hi = 0, n_train
        elif split == "val":
            self.lo, self.hi = n_train, n_blocks
        else:
            raise ValueError(split)

        self.data = full
        self.seq_len = seq_len
        self.eos_id = man["eos_id"]

    def __len__(self):
        return self.hi - self.lo

    def __getitem__(self, idx):
        b = self.lo + idx
        start = b * self.seq_len
        # Copy out of the memmap; cast uint16 -> int64 for embedding lookup.
        block = np.asarray(self.data[start:start + self.seq_len]).astype(np.int64)
        return torch.from_numpy(block)


class MixtureDataset(Dataset):
    """Interleave several PackedMemmapDatasets at fixed per-source ratios.

    Used for continual pretraining with *replay*: training a phase purely on a new
    corpus makes the model forget the old distribution, so we mix the old one back
    in. Ratios are by block (= by token, since every block is full).

    Sources are cycled independently, so a smaller source repeats rather than
    running out -- which is the intent of a fixed mixing ratio. Indices map to
    sources deterministically, so shuffling happens in the DataLoader as usual.
    """

    def __init__(self, datasets, weights=None):
        if not datasets:
            raise ValueError("need at least one dataset")
        self.datasets = list(datasets)
        w = list(weights) if weights else [1.0] * len(self.datasets)
        if len(w) != len(self.datasets):
            raise ValueError("weights must match datasets")
        tot = float(sum(w))
        if tot <= 0:
            raise ValueError("weights must sum to > 0")
        self.weights = [x / tot for x in w]
        # One "epoch" must be long enough that EVERY source is drawn at least len(d)
        # times, else the tail of an under-weighted source is unreachable forever
        # (the idx->position map is fixed, so extra epochs do not help). Using
        # sum(len(d)) silently capped UltraFineWeb/TinyStories at ~84% coverage.
        self.length = int(max(len(d) / w for d, w in zip(self.datasets, self.weights) if w > 0))
        # Deterministic interleave pattern of source ids at the requested ratios,
        # plus per-index rank so __getitem__ stays O(1) rather than rescanning.
        self._pattern = self._build_pattern()
        self._rank, self._per_cycle = self._index_pattern()
        self.seq_len = self.datasets[0].seq_len
        self.eos_id = self.datasets[0].eos_id

    def _build_pattern(self, resolution=1000):
        """Spread sources evenly at the target ratios (no long single-source runs),
        so any contiguous span of indices is already mixed."""
        used = [0] * len(self.datasets)
        out = []
        for k in range(resolution):
            # pick whichever source is furthest behind its target share so far
            best, best_gap = 0, float("-inf")
            for i, w in enumerate(self.weights):
                if w <= 0:
                    continue
                gap = w * (k + 1) - used[i]
                if gap > best_gap:
                    best, best_gap = i, gap
            used[best] += 1
            out.append(best)
        return out

    def _index_pattern(self):
        seen = [0] * len(self.datasets)
        rank = []
        for s in self._pattern:
            rank.append(seen[s])
            seen[s] += 1
        return rank, seen

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        j = idx % len(self._pattern)
        src = self._pattern[j]
        d = self.datasets[src]
        cycle = idx // len(self._pattern)
        pos = (cycle * self._per_cycle[src] + self._rank[j]) % len(d)
        return d[pos]
