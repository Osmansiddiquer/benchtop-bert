"""Time JEPA optimizer-steps at different micro-batch / grad-accum splits.

Every config below processes the SAME tokens per optimizer step (micro * accum * seq),
so the work is identical and any difference is pure hardware efficiency -- fewer, larger
kernels amortising weight reads better.

Read-only with respect to any running job: builds its own models, writes nothing.
Absolute numbers are inflated by contention if other jobs share the GPU; the RATIO
between configs is the meaningful output.
"""
import os
import sys

# Run as `python tools/x.py` from the repo root: sys.path[0] is tools/, so the
# package at the repo root is not importable without this.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

import torch
import torch.nn.functional as F

from mini_enc_transformer.training.pretrain import build_tokenizer, build_model
from mini_enc_transformer.training.jepa import Predictor, build_targets


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ckpt3/last.pt")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--tokens-per-step", type=int, default=32768)
    p.add_argument("--configs", default="16,32,48", help="micro-batch sizes to try")
    p.add_argument("--steps", type=int, default=3, help="optimizer steps per config")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


class C:
    d_model, d_k, d_v, n_heads, n_layers, d_embed = 768, 64, 64, 4, 4, 128


def time_config(micro, accum, seq, dev, tk_ids, steps):
    student = build_model(C(), tk_ids).to(dev)
    teacher = build_model(C(), tk_ids).to(dev)
    for p_ in teacher.parameters():
        p_.requires_grad_(False)
    teacher.eval()
    pred = Predictor(768).to(dev)
    params = list(student.parameters()) + list(pred.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4)

    x = torch.randint(100, 50000, (micro, seq), device=dev)
    mask = x.clone()
    mask[:, ::7] = 50280                                     # ~15% masked, shape only
    sel = torch.zeros_like(x, dtype=torch.bool); sel[:, ::7] = True

    def one_step():
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                with torch.no_grad():
                    _, th = teacher.encoder(x, output_hidden_states=True)
                    tgt = build_targets(th, 3)
                out = pred(student.encoder(mask))
                loss = F.smooth_l1_loss(out[sel], tgt[sel])
            (loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    one_step()                                               # warm up kernels/allocator
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        one_step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    peak = torch.cuda.max_memory_allocated() / 1e6 if dev.type == "cuda" else 0
    del student, teacher, pred, opt
    if dev.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return dt, peak


def main():
    a = parse_args()
    dev = torch.device(a.device)
    _, ids = build_tokenizer("allenai/OLMo-1B-hf")
    print(f"tokens/optimizer-step held constant at {a.tokens_per_step:,} (seq {a.seq_len})")
    print(f"{'micro':>6} {'accum':>6} {'s/step':>9} {'tok/s':>10} {'peak MB':>9} {'speedup':>8}")
    base = None
    for micro in [int(s) for s in a.configs.split(",")]:
        accum = a.tokens_per_step // (micro * a.seq_len)
        if accum < 1:
            print(f"{micro:>6} {'-':>6}  skipped (micro too large for this token budget)")
            continue
        try:
            dt, peak = time_config(micro, accum, a.seq_len, dev, ids, a.steps)
        except torch.cuda.OutOfMemoryError:
            print(f"{micro:>6} {accum:>6}  OOM (not enough free VRAM alongside running jobs)")
            torch.cuda.empty_cache()
            continue
        base = base or dt
        print(f"{micro:>6} {accum:>6} {dt:9.3f} {a.tokens_per_step/dt:10.0f} {peak:9.0f} "
              f"{base/dt:7.2f}x")
    print("\nSame work in every row -- only kernel sizes differ. Absolute times are")
    print("inflated by any concurrent job; compare the speedup column.")


if __name__ == "__main__":
    main()
