"""Fine-tune the pretrained encoder on SST-2 (sentiment) — headless version of
notebooks/finetune_sst2.ipynb, so the monitor can run it unattended once phase-2 finishes.

Loads the pretrained encoder (default checkpoints/ckpt2/last.pt, else checkpoints/ckpt/last.pt), drops the MLM head,
mean-pools the hidden states (no [CLS]), adds a linear head, and FULL fine-tunes on SST-2.
Reports validation accuracy (test labels are hidden) and writes results/finetune_sst2_results.json.
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mini_enc_transformer.model.encoder import Encoder
from mini_enc_transformer.training.pretrain import build_tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None, help="pretrained checkpoint (default: checkpoints/ckpt2/last.pt else checkpoints/ckpt/last.pt)")
    p.add_argument("--data", default="datasets/sst2")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-train-batches", type=int, default=None, help="cap batches/epoch (smoke test)")
    p.add_argument("--out", default="results/finetune_sst2_results.json")
    # Metrics go to a run folder so the live dashboard auto-discovers this run as a
    # tab (it globs */metrics.jsonl). Evaluating only at epoch end would give 3 points.
    p.add_argument("--run-dir", default="checkpoints/ckpt_sst2")
    p.add_argument("--eval-every", type=int, default=200, help="batches between val evals")
    p.add_argument("--log-every", type=int, default=20, help="batches between train-loss logs")
    # --- regularisation / schedule (the first run had none of this) ---
    p.add_argument("--warmup-frac", type=float, default=0.1, help="linear warmup fraction")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--llrd", type=float, default=1.0,
                   help="layer-wise LR decay; 1.0 disables. Lower layers get lr*llrd^depth. "
                        "Was 0.9, but a weight-drift probe showed the encoder moved only "
                        "0.5-3.4%% during fine-tuning -- under-adaptation, not forgetting -- "
                        "so the backbone is now left at full LR on this 4-layer model.")
    p.add_argument("--head-dropout", type=float, default=0.1)
    p.add_argument("--hidden-dropout", type=float, default=0.1,
                   help="dropout on encoder outputs before pooling (model itself has none)")
    p.add_argument("--seed", type=int, default=0)
    # Model selection must not touch the reported set: carve a dev split out of TRAIN
    # for best-checkpoint choice, and keep the official 872-example validation set for
    # the final number only.
    p.add_argument("--sel-frac", type=float, default=0.05,
                   help="fraction of train held out for model selection")
    return p.parse_args()


def lr_lambda(step, warmup, total):
    """Linear warmup then linear decay to 0 -- the standard BERT fine-tuning schedule.
    The first run used a constant LR, which let epochs 2-3 keep taking full-size steps
    on data the model had already fit."""
    if step < warmup:
        return step / max(1, warmup)
    return max(0.0, (total - step) / max(1, total - warmup))


def make_param_groups(model, lr, weight_decay, llrd, n_layers=4):
    """AdamW groups with (a) no weight decay on 1D params (LayerNorm gains, biases) --
    the first run decayed them by mistake -- and (b) layer-wise LR decay, so lower
    layers move less than the head."""
    def depth_of(name):
        if name.startswith("head") or name.startswith("drop"):
            return 0
        if name.startswith("encoder.final_norm"):
            return 1
        if name.startswith("encoder.encoder_blocks."):
            return n_layers + 1 - int(name.split(".")[2])
        return n_layers + 2                      # embeddings: slowest
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        d = depth_of(name)
        decay = p.ndim >= 2
        key = (d, decay)
        groups.setdefault(key, {"params": [], "lr": lr * (llrd ** d),
                                "weight_decay": weight_decay if decay else 0.0})
        groups[key]["params"].append(p)
    return list(groups.values())


class EncoderForSentiment(nn.Module):
    def __init__(self, ckpt_path, ids, n_classes=2, head_dropout=0.1, hidden_dropout=0.1,
                 d_model=768, d_embed=128, n_heads=4, n_layers=4, d_k=64, d_v=64):
        super().__init__()
        self.encoder = Encoder(ids["vocab_size"], d_model, d_k, d_v, n_heads, n_layers,
                               pad_id=ids["pad_id"], d_embed=d_embed)
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        if any(k.startswith("head.") for k in sd):
            # A saved fine-tune checkpoint (encoder + head): load the whole thing.
            self.drop = nn.Dropout(head_dropout)
            self.hdrop = nn.Dropout(hidden_dropout)
            self.head = nn.Linear(d_model, n_classes)
            self.load_state_dict(sd)
        else:
            enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
            self.encoder.load_state_dict(enc)       # only the encoder subtree; MLM head dropped
            self.drop = nn.Dropout(head_dropout)
            # The encoder has no dropout of its own; add it here at fine-tune time,
            # where 67K labelled examples against 28.7M params flips the regime.
            self.hdrop = nn.Dropout(hidden_dropout)
            self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, mask):
        H = self.hdrop(self.encoder(x))             # (B,T,768) hidden states, NOT token logits
        w = mask.unsqueeze(-1).float()
        pooled = (H * w).sum(1) / w.sum(1).clamp(min=1)   # mean-pool over real tokens (no [CLS])
        return self.head(self.drop(pooled))


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    # Prefer a saved fine-tune checkpoint if one exists (warm start), else the
    # pretrained backbone. The first run saved nothing, so there was none to resume.
    ft = os.path.join(a.run_dir, "best.pt")
    ckpt = a.ckpt or (ft if os.path.exists(ft) else
                      ("checkpoints/ckpt2/last.pt" if os.path.exists("checkpoints/ckpt2/last.pt") else "checkpoints/ckpt/last.pt"))
    print(f"[finetune] backbone={ckpt} device={a.device}", flush=True)

    tk, ids = build_tokenizer("allenai/OLMo-1B-hf")
    from datasets import load_from_disk
    ds = load_from_disk(a.data)
    train_ds, val_ds = ds["train"], ds["validation"]

    def encode(b):
        return tk(b["sentence"], truncation=True, max_length=a.max_len)
    train_tok = train_ds.map(encode, batched=True)
    val_tok = val_ds.map(encode, batched=True)

    def collate(rows):
        seqs = [r["input_ids"] for r in rows]
        L = max(len(s) for s in seqs)
        x = torch.full((len(rows), L), ids["pad_id"], dtype=torch.long)
        m = torch.zeros((len(rows), L), dtype=torch.bool)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s); m[i, :len(s)] = True
        y = torch.tensor([r["label"] for r in rows])
        return x, m, y

    # Model selection happens on a split carved out of TRAIN, never on the official
    # 872-example validation set -- selecting and reporting on the same tiny set is
    # what inflated the previous headline (max over 34 evals, ~1-2 SE optimistic).
    n_sel = int(len(train_tok) * a.sel_frac)
    split = train_tok.train_test_split(test_size=n_sel, seed=a.seed)
    train_tok, sel_tok = split["train"], split["test"]
    print(f"[finetune] train={len(train_tok)} selection={len(sel_tok)} "
          f"report_val={len(val_tok)}", flush=True)

    train_loader = DataLoader(train_tok, batch_size=a.batch, shuffle=True, collate_fn=collate)
    sel_loader = DataLoader(sel_tok, batch_size=64, shuffle=False, collate_fn=collate)
    val_loader = DataLoader(val_tok, batch_size=64, shuffle=False, collate_fn=collate)

    model = EncoderForSentiment(ckpt, ids, head_dropout=a.head_dropout,
                                hidden_dropout=a.hidden_dropout).to(a.device)
    opt = torch.optim.AdamW(
        make_param_groups(model, a.lr, a.weight_decay, a.llrd), lr=a.lr)
    total_steps = a.epochs * (a.max_train_batches or len(train_loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, int(a.warmup_frac * total_steps), total_steps))
    lossf = nn.CrossEntropyLoss()

    os.makedirs(a.run_dir, exist_ok=True)
    metrics_path = os.path.join(a.run_dir, "metrics.jsonl")
    n_params = sum(p.numel() for p in model.parameters())
    steps_per_epoch = a.max_train_batches or len(train_loader)

    def append_metric(obj):
        with open(metrics_path, "a") as f:
            f.write(json.dumps(obj) + "\n")
            f.flush()

    append_metric({"t": "meta", "max_steps": a.epochs * steps_per_epoch, "start_step": 0,
                   "params_M": round(n_params / 1e6, 1), "eff_batch": a.batch,
                   "tokens_per_step": a.batch * a.max_len, "vocab": ids["vocab_size"],
                   "dataset": "sst2", "lr": a.lr, "time": time.time()})

    @torch.no_grad()
    def evaluate(loader):
        """Accuracy + cross-entropy. Keys match train.py so the dashboard charts it
        unchanged -- `masked_acc` here means classification accuracy."""
        model.eval(); c = t = 0; tot = 0.0; nb = 0
        for x, m, y in loader:
            x, m, y = x.to(a.device), m.to(a.device), y.to(a.device)
            logits = model(x, m)
            tot += lossf(logits, y).item(); nb += 1
            c += (logits.argmax(-1) == y).sum().item(); t += y.numel()
        model.train(); return c / t, tot / max(1, nb)

    best_sel = 0.0
    best_step = 0
    hist = []
    step = 0
    running = 0.0
    nrun = 0
    ckpt_best = os.path.join(a.run_dir, "best.pt")

    def save_best():
        """Atomic, so a sleep mid-write cannot corrupt the checkpoint. The previous
        run saved nothing at all and its weights were lost."""
        tmp = ckpt_best + ".tmp"
        torch.save({"model": model.state_dict(), "step": step, "sel_acc": best_sel,
                    "backbone": ckpt, "args": vars(a)}, tmp)
        os.replace(tmp, ckpt_best)

    for ep in range(a.epochs):
        for i, (x, m, y) in enumerate(train_loader):
            if a.max_train_batches and i >= a.max_train_batches:
                break
            x, m, y = x.to(a.device), m.to(a.device), y.to(a.device)
            loss = lossf(model(x, m), y)
            opt.zero_grad(); loss.backward()
            if a.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt.step(); sched.step()
            step += 1; running += loss.item(); nrun += 1

            if step % a.log_every == 0:
                append_metric({"t": "train", "step": step, "loss": round(running / nrun, 4),
                               "lr": sched.get_last_lr()[0], "epoch": ep + 1, "time": time.time()})
                running = 0.0; nrun = 0

            if step % a.eval_every == 0:
                sacc, sl = evaluate(sel_loader)
                if sacc > best_sel:
                    best_sel, best_step = sacc, step
                    save_best()
                append_metric({"t": "eval", "step": step, "val_loss": round(sl, 4),
                               "masked_acc": round(sacc, 4), "time": time.time()})
                print(f"[finetune] step {step} sel_acc={sacc:.4f} sel_loss={sl:.4f} "
                      f"(best {best_sel:.4f} @ {best_step})", flush=True)

        sacc, sl = evaluate(sel_loader)
        if sacc > best_sel:
            best_sel, best_step = sacc, step
            save_best()
        hist.append(sacc)
        append_metric({"t": "eval", "step": step, "val_loss": round(sl, 4),
                       "masked_acc": round(sacc, 4), "time": time.time()})
        print(f"[finetune] epoch {ep+1}/{a.epochs}: sel_acc={sacc:.4f} "
              f"(best {best_sel:.4f})", flush=True)
        with open(a.out, "w") as f:
            json.dump({"backbone": ckpt, "sel_acc_per_epoch": hist,
                       "best_sel_acc": best_sel, "best_step": best_step,
                       "epochs_done": ep + 1, "epochs": a.epochs, "lr": a.lr,
                       "batch": a.batch, "complete": False}, f, indent=2)

    # Final, honest number: restore the checkpoint chosen on the selection split and
    # evaluate the official validation set ONCE. No max-over-evals, no peeking.
    if os.path.exists(ckpt_best):
        model.load_state_dict(torch.load(ckpt_best, map_location=a.device)["model"])
    val_acc, val_loss = evaluate(val_loader)
    n = len(val_tok)
    se = (val_acc * (1 - val_acc) / n) ** 0.5
    res = {"backbone": ckpt, "sel_acc_per_epoch": hist, "best_sel_acc": best_sel,
           "best_step": best_step, "val_acc": val_acc, "val_loss": val_loss,
           "val_n": n, "val_se": se, "epochs": a.epochs, "lr": a.lr, "llrd": a.llrd,
           "batch": a.batch, "checkpoint": ckpt_best, "complete": True}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[finetune] DONE val_acc={val_acc:.4f} +/- {se:.4f} (n={n}) "
          f"| selected step {best_step} (sel_acc {best_sel:.4f}) | ckpt {ckpt_best}",
          flush=True)


if __name__ == "__main__":
    main()
