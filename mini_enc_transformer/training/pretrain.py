"""MLM pretraining loop for the from-scratch BERT, sized for a 4GB laptop GPU.

Reads the local packed memmap (see data_prep.py / data.py), applies BERT 80/10/10
masking on the fly, and trains the factorized-embedding encoder with AdamW +
warmup/cosine under bf16 autocast, gradient accumulation, and gradient clipping.

Built to survive laptop sleep and interruption: it writes an atomic checkpoint
(model + optimizer + scheduler + step + RNG) every --ckpt-every steps and, with
--resume, continues from checkpoints/ckpt/last.pt. Training touches only the local memmap, so
it is completely independent of the network once data is prepped.

Importable: `build_tokenizer`, `build_model`, `train(cfg)`. Also a CLI.
"""
import argparse
import json
import math
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from mini_enc_transformer.data.dataset import MixtureDataset, PackedMemmapDataset
from mini_enc_transformer.model.mlm import BertForMaskedLM, mask_tokens, IGNORE_INDEX


def parse_args():
    p = argparse.ArgumentParser()
    # model
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--d-embed", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-k", type=int, default=64)
    p.add_argument("--d-v", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=128)
    # data
    p.add_argument("--data-dir", default="data")
    p.add_argument("--data-name", default="ultrafineweb_en",
                   help="corpus name, or a replay mixture: 'a:0.5,b:0.5' (weights by token)")
    p.add_argument("--limit-tokens", type=int, default=None)
    # optim
    # micro-batch 16 -> ~1.8GB peak on the 4GB 3050 Ti (measured), leaving room for
    # the desktop; 32 -> ~3.3GB works if the GPU is otherwise idle. accum makes up
    # the effective batch (16*16 = 256).
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=16)  # effective batch = micro*accum
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-frac", type=float, default=0.06)
    p.add_argument("--max-steps", type=int, default=10000)  # optimizer steps (sets cosine horizon)
    p.add_argument("--max-seconds", type=float, default=None,
                   help="wall-clock training budget; stops + checkpoints when reached")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--mlm-prob", type=float, default=0.15)
    # infra
    p.add_argument("--ckpt-dir", default="ckpt")
    p.add_argument("--ckpt-every", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init-from", default=None,
                   help="warm-start MODEL weights from this checkpoint, but start a fresh "
                        "optimizer/schedule/step-0 run (for continued pretraining on new data)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tokenizer", default="allenai/OLMo-1B-hf")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_tokenizer(name):
    """OLMo tokenizer + an added [MASK] (OLMo ships none). Returns (tk, ids dict)."""
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(name)
    if tk.mask_token is None:
        tk.add_special_tokens({"mask_token": "[MASK]"})
    ids = {
        "vocab_size": len(tk),
        "mask_id": tk.mask_token_id,
        "pad_id": tk.pad_token_id,
        # protect structural tokens from being masked/scored
        "special_ids": tuple(i for i in tk.all_special_ids if i != tk.mask_token_id),
    }
    return tk, ids


def build_model(cfg, ids):
    return BertForMaskedLM(
        vocab_size=ids["vocab_size"], d_model=cfg.d_model, d_k=cfg.d_k, d_v=cfg.d_v,
        n_heads=cfg.n_heads, n_layers=cfg.n_layers, pad_id=ids["pad_id"], d_embed=cfg.d_embed,
    )


def make_optimizer(model, lr, weight_decay):
    # Decay 2D weights (matmuls); skip 1D (biases, LayerNorm gains). Standard nanoGPT split.
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    nodecay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.98), eps=1e-6,
    )


def lr_lambda(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))  # cosine -> 0


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def append_metric(path, obj):
    """Append one JSON record + flush, so the live dashboard sees it immediately."""
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()


@torch.no_grad()
def evaluate(model, val_loader, ids, device, max_batches=20):
    model.eval()
    tot_loss, tot_correct, tot_scored = 0.0, 0, 0
    n = 0
    for block in val_loader:
        if n >= max_batches:
            break
        n += 1
        masked, labels = mask_tokens(block, ids["mask_id"], ids["vocab_size"],
                                     special_token_ids=ids["special_ids"], pad_token_id=ids["pad_id"])
        masked, labels = masked.to(device), labels.to(device)
        out = model(masked, labels)
        tot_loss += out["loss"].item()
        scored = labels != IGNORE_INDEX
        pred = out["logits"].argmax(-1)
        tot_correct += (pred[scored] == labels[scored]).sum().item()
        tot_scored += scored.sum().item()
    model.train()
    return tot_loss / max(1, n), tot_correct / max(1, tot_scored)


def train(cfg):
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    device = cfg.device
    use_amp = device == "cuda"

    tk, ids = build_tokenizer(cfg.tokenizer)
    model = build_model(cfg, ids).to(device)
    opt = make_optimizer(model, cfg.lr, cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, int(cfg.warmup_frac * cfg.max_steps), cfg.max_steps))

    def build_ds(split):
        """Spec: 'name' | 'a:w,b:w' | 'a:w:limit,b:w:limit'.

        weight = share of sampled tokens; limit = cap on tokens used from that corpus
        (so you can take, say, exactly half of one and traverse it exactly once).
        """
        specs = []
        for part in [s.strip() for s in cfg.data_name.split(",") if s.strip()]:
            bits = part.split(":")
            name = bits[0]
            w = float(bits[1]) if len(bits) > 1 and bits[1] else 1.0
            lim = int(float(bits[2])) if len(bits) > 2 and bits[2] else cfg.limit_tokens
            specs.append((name, w, lim))
        dss = [PackedMemmapDataset(cfg.data_dir, n, cfg.seq_len, split, limit_tokens=lim)
               for n, _, lim in specs]
        if len(dss) == 1:
            return dss[0]
        return MixtureDataset(dss, [w for _, w, _ in specs])

    train_ds = build_ds("train")
    val_ds = build_ds("val")
    g = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.micro_batch, shuffle=True,
                              drop_last=True, generator=g, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.micro_batch, shuffle=False, drop_last=False)

    start_step = 0
    last_path = os.path.join(cfg.ckpt_dir, "last.pt")
    resuming = cfg.resume and os.path.exists(last_path)
    if getattr(cfg, "init_from", None) and not resuming:
        # Continued pretraining: load only the weights, keep a fresh optimizer +
        # LR schedule + step counter so the new data gets its own warmup/cosine.
        ick = torch.load(cfg.init_from, map_location=device)
        model.load_state_dict(ick["model"])
        print(f"[init] warm-started weights from {cfg.init_from} "
              f"(fresh optimizer/schedule, step 0)", flush=True)
    if resuming:
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_step = ck["step"]
        # RNG states must be CPU ByteTensors; map_location=cuda moved them to GPU.
        torch.set_rng_state(ck["rng"].cpu().to(torch.uint8))
        if use_amp and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu().to(torch.uint8))
        print(f"[resume] from step {start_step}", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params/1e6:.1f}M params | vocab {ids['vocab_size']} | "
          f"device {device} | train blocks {len(train_ds):,} | eff batch "
          f"{cfg.micro_batch*cfg.grad_accum}", flush=True)

    # Structured metrics for the live dashboard (one JSON line per event).
    metrics_path = os.path.join(cfg.ckpt_dir, "metrics.jsonl")
    tokens_per_step = cfg.micro_batch * cfg.grad_accum * cfg.seq_len
    append_metric(metrics_path, {"t": "meta", "max_steps": cfg.max_steps,
                                 "start_step": start_step, "params_M": round(n_params / 1e6, 1),
                                 "eff_batch": cfg.micro_batch * cfg.grad_accum,
                                 "tokens_per_step": tokens_per_step, "vocab": ids["vocab_size"],
                                 "dataset": cfg.data_name, "lr": cfg.lr, "time": time.time()})

    def save_ckpt():
        atomic_save({"model": model.state_dict(), "opt": opt.state_dict(),
                     "sched": sched.state_dict(), "step": step, "rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state() if use_amp else None,
                     "cfg": vars(cfg), "ids": ids}, last_path)

    model.train()
    step = start_step
    micro = 0
    opt.zero_grad(set_to_none=True)
    data_iter = iter(train_loader)
    running = 0.0
    t_start = time.time()
    max_seconds = getattr(cfg, "max_seconds", None)
    stop = False

    while step < cfg.max_steps and not stop:
        try:
            block = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            block = next(data_iter)

        masked, labels = mask_tokens(block, ids["mask_id"], ids["vocab_size"],
                                     special_token_ids=ids["special_ids"], pad_token_id=ids["pad_id"])
        masked, labels = masked.to(device), labels.to(device)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(masked, labels)["loss"]
        else:
            loss = model(masked, labels)["loss"]

        (loss / cfg.grad_accum).backward()
        running += loss.item()
        micro += 1

        if micro == cfg.grad_accum:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            micro = 0

            if step % cfg.log_every == 0:
                avg = running / (cfg.log_every * cfg.grad_accum)
                running = 0.0
                lr_now = sched.get_last_lr()[0]
                print(f"step {step:6d}/{cfg.max_steps} | loss {avg:6.3f} | lr {lr_now:.2e}", flush=True)
                append_metric(metrics_path, {"t": "train", "step": step, "loss": round(avg, 4),
                                             "lr": lr_now, "tokens": step * tokens_per_step,
                                             "time": time.time()})
            if step % cfg.eval_every == 0:
                vl, va = evaluate(model, val_loader, ids, device)
                print(f"  [eval] step {step} | val_loss {vl:.3f} | masked_acc {va:.3f}", flush=True)
                append_metric(metrics_path, {"t": "eval", "step": step, "val_loss": round(vl, 4),
                                             "masked_acc": round(va, 4), "time": time.time()})
            if step % cfg.ckpt_every == 0:
                save_ckpt()
            if max_seconds is not None and time.time() - t_start >= max_seconds:
                print(f"[budget] hit {max_seconds:.0f}s wall-clock at step {step}", flush=True)
                stop = True

    save_ckpt()
    print(f"[done] trained to step {step} in {(time.time()-t_start)/60:.1f} min; "
          f"checkpoint at {last_path}", flush=True)


if __name__ == "__main__":
    train(parse_args())
