"""Mild CPU fine-tune: teach the MLM to CONTINUE text (and to stop).

Training shape == inference shape: left context + 3 [MASK], supervise ONLY the first
mask slot with the true next token. EOS targets are oversampled so the model learns
where documents end. Everything else about the model is left alone.
"""
import numpy as np, torch, time, json, argparse, os
from mini_enc_transformer.training.pretrain import build_tokenizer, build_model
from mini_enc_transformer.model.mlm import IGNORE_INDEX

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=300)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--eos-frac", type=float, default=0.30, help="fraction of examples whose target is EOS")
p.add_argument("--threads", type=int, default=4)
p.add_argument("--warmup-frac", type=float, default=0.10)
p.add_argument("--decay-frac", type=float, default=0.30, help="WSD: final fraction spent decaying to 0")
p.add_argument("--min-ctx", type=int, default=8, help="shortest sampled context (matches short prompts)")
p.add_argument("--corpus", default="tinystories")
p.add_argument("--ckpt", default="checkpoints/ckpt3/last.pt")
p.add_argument("--out", default="checkpoints/ckpt_autoreg/ckpt_autoreg.pt")
a = p.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(0); np.random.seed(0)

class C: d_model, d_k, d_v, n_heads, n_layers, d_embed = 768, 64, 64, 4, 4, 128
SEQ, NM = 128, 3
tk, ids = build_tokenizer("allenai/OLMo-1B-hf")
EOS = tk.eos_token_id
model = build_model(C(), ids)
model.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"]); model.train()

man = json.load(open(f"data/{a.corpus}.manifest.json"))
data = np.memmap(f"data/{a.corpus}.bin", dtype=np.uint16, mode="r", shape=(man["target_tokens"],))
N = man["tokens_written"]
print(f"corpus {a.corpus}: {N:,} tokens", flush=True)

# Positions where the NEXT token is EOS -> these teach termination.
scan = np.asarray(data[:min(N, 40_000_000)])
eos_at = np.flatnonzero(scan == EOS)
eos_at = eos_at[(eos_at > SEQ) & (eos_at < len(scan) - 1)]
print(f"found {len(eos_at):,} document boundaries to sample from", flush=True)

def batch(bs, ctx_len):
    """One context length per batch, so examples stack without padding.

    Context length is randomised because at inference the window GROWS: the first
    generated token sees only the prompt, later ones see the full 125. Training only
    at full length mismatches every short prompt. The model has no key-padding mask,
    so left-padding is not an option -- varying the length per batch is.
    """
    xs, ys = [], []
    for _ in range(bs):
        if np.random.rand() < a.eos_frac and len(eos_at):
            tgt = int(np.random.choice(eos_at))            # next token IS eos
        else:
            tgt = int(np.random.randint(SEQ, min(N, len(scan)) - 1))
        assert tgt - ctx_len >= 0
        ctx = np.asarray(scan[tgt - ctx_len:tgt]).astype(np.int64)
        xs.append(np.concatenate([ctx, np.full(NM, ids["mask_id"], dtype=np.int64)]))
        lab = np.full(ctx_len + NM, IGNORE_INDEX, dtype=np.int64)
        lab[ctx_len] = int(scan[tgt])                      # supervise slot 1 only
        ys.append(lab)
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))

def wsd(step, total, wf, df):
    """Warmup-Stable-Decay (MiniCPM 2024): ramp, hold flat, then decay only at the end.
    Unlike cosine-to-zero it leaves the run extendable -- you can stop after the stable
    phase and continue later without paying a re-warm spike, which is exactly the tax
    phases 2 and 3 of this project paid."""
    w, d = int(wf * total), int(df * total)
    if step < w:
        return step / max(1, w)
    if step < total - d:
        return 1.0
    return max(0.0, (total - step) / max(1, d))

opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: wsd(s, a.steps, a.warmup_frac, a.decay_frac))
t0 = time.time(); run = 0.0
for s in range(1, a.steps + 1):
    L = int(np.random.randint(a.min_ctx, SEQ - NM + 1))    # randomised context per batch
    x, y = batch(a.batch, L)
    loss = model(x, y)["loss"]
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    run += loss.item()
    if s % 25 == 0:
        print(f"step {s}/{a.steps} loss {run/25:.4f} lr {sched.get_last_lr()[0]:.2e} "
              f"({(time.time()-t0)/s:.2f}s/step)", flush=True); run = 0.0
torch.save({"model": model.state_dict(), "base": a.ckpt, "steps": a.steps,
            "lr": a.lr, "eos_frac": a.eos_frac}, a.out)
print(f"saved {a.out} in {(time.time()-t0)/60:.1f} min", flush=True)
