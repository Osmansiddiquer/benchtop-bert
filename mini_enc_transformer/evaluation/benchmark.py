"""Comprehensive checkpoint evaluation: linear probe vs full fine-tune, SST-2 and STS-B.

Why two protocols. Full fine-tuning lets the encoder reorganise itself, which can mask
differences in the representation you started from -- two very different checkpoints can
converge to similar accuracy given enough gradient steps. A **linear probe** freezes the
encoder and trains only a linear head, so it measures what the representation *already
encodes*. If a pretraining change improved semantic structure, the probe should show a
larger gap than the fine-tuned number does.

Why two tasks. SST-2 sentiment is substantially lexical -- "brilliant", "tedious" carry
most of the signal -- so a model can score well without deep semantics. STS-B asks
whether two sentences *mean* the same thing, which is a far more direct test.

STS-B also supports a zero-shot mode: cosine similarity between mean-pooled sentence
embeddings, correlated against the human scores, with no training at all. That is the
purest available measure of semantic structure in the frozen representation.

Nothing is written to disk except the results JSON -- no fine-tuned checkpoints are saved.

    python -m mini_enc_transformer.evaluation.benchmark --ckpt checkpoints/ckpt3/last.pt
"""
import argparse
import copy
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_enc_transformer.arch import load_checkpoint_encoder
from mini_enc_transformer.training.pretrain import build_tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tasks", default="sst2,stsb")
    p.add_argument("--modes", default="probe,finetune", help="probe = frozen encoder")
    p.add_argument("--epochs-probe", type=int, default=8, help="cheap: encoder is frozen")
    p.add_argument("--epochs-finetune", type=int, default=3)
    p.add_argument("--lr-probe", type=float, default=1e-3)
    p.add_argument("--lr-finetune", type=float, default=3e-5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--sel-frac", type=float, default=0.05,
                   help="held-out slice of TRAIN used for model selection, so the "
                        "reported dev set is touched exactly once")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--d-embed", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-k", type=int, default=64)
    p.add_argument("--d-v", type=int, default=64)
    return p.parse_args()


# ---------------------------------------------------------------- model
def load_encoder(cfg, ids):
    """Architecture is DETECTED, not taken from cfg: --n-layers/--d-ff defaults describe
    v1, and silently building a v1 shape for a v2 checkpoint fails with nothing but a
    wall of size mismatches."""
    enc, arch, _ = load_checkpoint_encoder(
        cfg.ckpt, ids, d_model=cfg.d_model, d_k=cfg.d_k, d_v=cfg.d_v,
        n_heads=cfg.n_heads, d_embed=cfg.d_embed)
    print(f"[bench] {cfg.ckpt}: {arch['n_layers']}L d_ff={arch['d_ff']} "
          f"{arch['act'].upper()}", flush=True)
    return enc


def mean_pool(h, mask):
    w = mask.unsqueeze(-1).float()
    return (h * w).sum(1) / w.sum(1).clamp(min=1)


class SentenceHead(nn.Module):
    """Single-sentence classification: mean-pool -> dropout -> linear."""

    def __init__(self, encoder, d_model, n_out, frozen):
        super().__init__()
        self.encoder, self.frozen = encoder, frozen
        # Feature norm before the linear head. With a frozen encoder the head cannot
        # rescale its inputs, so unnormalised features make the probe unstable.
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(d_model, n_out)

    def forward(self, x, m):
        if self.frozen:
            with torch.no_grad():
                h = self.encoder(x)
        else:
            h = self.encoder(x)
        return self.head(self.drop(self.norm(mean_pool(h, m))))


class PairHead(nn.Module):
    """Sentence-pair regression. Features [u, v, |u-v|, u*v] are the Sentence-BERT
    combination -- the difference and product terms are what let a *linear* head express
    similarity, which a plain concatenation cannot."""

    def __init__(self, encoder, d_model, frozen):
        super().__init__()
        self.encoder, self.frozen = encoder, frozen
        # The u*v term is an order of magnitude larger than u or |u-v|; without a norm
        # it dominates and a frozen-feature probe fits an inverted mapping (measured
        # spearman -0.44 before this was added).
        self.norm = nn.LayerNorm(4 * d_model)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(4 * d_model, 1)

    def embed(self, x, m):
        if self.frozen:
            with torch.no_grad():
                h = self.encoder(x)
        else:
            h = self.encoder(x)
        return mean_pool(h, m)

    def forward(self, x1, m1, x2, m2):
        u, v = self.embed(x1, m1), self.embed(x2, m2)
        f = self.norm(torch.cat([u, v, (u - v).abs(), u * v], -1))
        return self.head(self.drop(f)).squeeze(-1)


# ---------------------------------------------------------------- metrics
def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))


def spearman(a, b):
    def rank(t):
        idx = t.argsort()
        r = torch.empty_like(idx, dtype=torch.float)
        r[idx] = torch.arange(len(t), dtype=torch.float)
        return r
    return pearson(rank(a), rank(b))


def acc_se(acc, n):
    return math.sqrt(max(acc * (1 - acc), 1e-12) / n)


def corr_se(r, n):
    """Fisher-z standard error, mapped back to correlation units."""
    if n <= 3:
        return float("nan")
    se_z = 1.0 / math.sqrt(n - 3)
    lo, hi = math.tanh(math.atanh(max(min(r, 0.999), -0.999)) - se_z), \
             math.tanh(math.atanh(max(min(r, 0.999), -0.999)) + se_z)
    return (hi - lo) / 2


# ---------------------------------------------------------------- data
def make_collate(ids, tk, max_len, pair):
    def enc(texts):
        seqs = [tk(t, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
                or [ids["pad_id"]] for t in texts]
        L = max(len(s) for s in seqs)
        x = torch.full((len(seqs), L), ids["pad_id"], dtype=torch.long)
        m = torch.zeros((len(seqs), L), dtype=torch.bool)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s); m[i, :len(s)] = True
        return x, m

    def collate(rows):
        y = torch.tensor([r["label"] for r in rows], dtype=torch.float if pair else torch.long)
        if pair:
            x1, m1 = enc([r["sentence1"] for r in rows])
            x2, m2 = enc([r["sentence2"] for r in rows])
            return (x1, m1, x2, m2), y
        x, m = enc([r["sentence"] for r in rows])
        return (x, m), y
    return collate


# ---------------------------------------------------------------- eval loops
@torch.no_grad()
def predict(model, loader, device, pair):
    model.eval()
    P, Y = [], []
    for xs, y in loader:
        xs = [t.to(device) for t in xs]
        P.append(model(*xs).float().cpu()); Y.append(y)
    model.train()
    return torch.cat(P), torch.cat(Y)


def score(pred, y, pair):
    if pair:
        r, rho = pearson(pred, y), spearman(pred, y)
        return {"pearson": r, "spearman": rho, "se": corr_se(rho, len(y)),
                "primary": rho, "metric": "spearman"}
    acc = (pred.argmax(-1) == y.long()).float().mean().item()
    return {"accuracy": acc, "se": acc_se(acc, len(y)), "primary": acc, "metric": "accuracy"}


def run_task(cfg, ids, tk, task, mode, log):
    from datasets import load_from_disk
    pair = task == "stsb"
    ds = load_from_disk("datasets/stsb" if pair else "datasets/sst2")
    train, dev = ds["train"], ds["validation"]

    n_sel = max(1, int(len(train) * cfg.sel_frac))
    sp = train.train_test_split(test_size=n_sel, seed=cfg.seed)
    train, sel = sp["train"], sp["test"]

    coll = make_collate(ids, tk, cfg.max_len, pair)
    tl = DataLoader(train, batch_size=cfg.batch, shuffle=True, collate_fn=coll)
    sl = DataLoader(sel, batch_size=64, collate_fn=coll)
    dl = DataLoader(dev, batch_size=64, collate_fn=coll)

    frozen = mode == "probe"
    enc = load_encoder(cfg, ids)
    if frozen:
        for p_ in enc.parameters():
            p_.requires_grad_(False)
        enc.eval()
    model = (PairHead(enc, cfg.d_model, frozen) if pair
             else SentenceHead(enc, cfg.d_model, 2, frozen)).to(cfg.device)

    epochs = cfg.epochs_probe if frozen else cfg.epochs_finetune
    lr = cfg.lr_probe if frozen else cfg.lr_finetune
    trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    total = epochs * len(tl)
    warm = int(0.1 * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    lossf = nn.MSELoss() if pair else nn.CrossEntropyLoss()

    best_sel, best_state = -1e9, None
    for ep in range(epochs):
        for xs, y in tl:
            xs = [t.to(cfg.device) for t in xs]; y = y.to(cfg.device)
            out = model(*xs)
            loss = lossf(out, y if pair else y.long())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step()
        s = score(*predict(model, sl, cfg.device, pair), pair)
        if s["primary"] > best_sel:
            best_sel = s["primary"]
            # Selection state is held in MEMORY only; nothing is written to disk.
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        log(f"    {task}/{mode} epoch {ep+1}/{epochs}: sel {s['metric']} {s['primary']:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    res = score(*predict(model, dl, cfg.device, pair), pair)
    res.update({"mode": mode, "task": task, "n_dev": len(dev), "sel_best": best_sel,
                "epochs": epochs, "lr": lr})
    return res


@torch.no_grad()
def stsb_zero_shot(cfg, ids, tk, log):
    """Cosine similarity of mean-pooled embeddings vs human scores. No training at all --
    the purest read on whether the frozen representation encodes meaning."""
    from datasets import load_from_disk
    dev = load_from_disk("datasets/stsb")["validation"]
    enc = load_encoder(cfg, ids).to(cfg.device).eval()
    coll = make_collate(ids, tk, cfg.max_len, True)
    dl = DataLoader(dev, batch_size=64, collate_fn=coll)
    S, Y = [], []
    for (x1, m1, x2, m2), y in dl:
        x1, m1, x2, m2 = x1.to(cfg.device), m1.to(cfg.device), x2.to(cfg.device), m2.to(cfg.device)
        u, v = mean_pool(enc(x1), m1), mean_pool(enc(x2), m2)
        S.append(F.cosine_similarity(u, v, dim=-1).float().cpu()); Y.append(y)
    S, Y = torch.cat(S), torch.cat(Y)
    r, rho = pearson(S, Y), spearman(S, Y)
    log(f"    stsb/zero-shot: spearman {rho:.4f} pearson {r:.4f}")
    return {"task": "stsb", "mode": "zero-shot-cosine", "pearson": r, "spearman": rho,
            "se": corr_se(rho, len(Y)), "primary": rho, "metric": "spearman", "n_dev": len(Y)}


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    tk, ids = build_tokenizer("allenai/OLMo-1B-hf")
    t0 = time.time()

    def log(m):
        print(m, flush=True)

    log(f"[benchmark] {cfg.ckpt}  device={cfg.device}")
    results = [stsb_zero_shot(cfg, ids, tk, log)] if "stsb" in cfg.tasks else []
    for task in [t for t in cfg.tasks.split(",") if t.strip()]:
        for mode in [m for m in cfg.modes.split(",") if m.strip()]:
            results.append(run_task(cfg, ids, tk, task, mode, log))

    print(f"\n{'task':6s} {'mode':18s} {'metric':10s} {'value':>8s} {'+/-':>7s}")
    print("-" * 54)
    for r in results:
        print(f"{r['task']:6s} {r['mode']:18s} {r['metric']:10s} "
              f"{r['primary']:8.4f} {r.get('se', float('nan')):7.4f}")

    out = cfg.out or f"results/benchmark_{os.path.basename(os.path.dirname(cfg.ckpt))}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"ckpt": cfg.ckpt, "results": results,
                   "minutes": round((time.time() - t0) / 60, 1)}, f, indent=2)
    print(f"\n-> {out}  ({(time.time()-t0)/60:.1f} min).  No checkpoints were saved.")


if __name__ == "__main__":
    main()
