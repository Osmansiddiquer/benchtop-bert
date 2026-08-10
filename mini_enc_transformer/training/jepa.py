"""data2vec / JEPA-style pretraining: predict latent representations, not tokens.

The MLM objective asks "which token was here?", which spends capacity on surface form:
p(table)=0.25 vs p(dog)=0.25 are equally wrong to cross-entropy even though one is far
closer in meaning. A latent objective instead asks the student to predict what a
full-context encoder *computes* at the masked positions, so being semantically close
counts as being close.

    student f_theta : sees the sequence with spans replaced by [MASK]
    teacher f_xi    : EMA of the student, sees the FULL sequence, no gradient
    predictor g_phi : maps student hidden states at masked positions -> target space
    loss            : smooth-L1( g_phi(h_student), normalise(top-K teacher layers) )

Both encoders are warm-started from the same checkpoint, so the teacher produces
meaningful targets from step 1 rather than having to bootstrap from noise.

COLLAPSE is the failure mode to watch. Nothing stops the model emitting a constant
vector and driving the loss to zero -- unlike cross-entropy, whose labels are fixed.
The EMA teacher + stop-gradient is the structural defence; the telemetry below
(target std, effective rank, cosine to the batch mean) is how we detect it anyway.
There is no masked-token accuracy here, by construction: the only real verdict is the
downstream SST-2 number.
"""
import argparse
import copy
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from mini_enc_transformer.data.dataset import MixtureDataset, PackedMemmapDataset
from mini_enc_transformer.model.mlm import IGNORE_INDEX, mask_tokens
from mini_enc_transformer.training.pretrain import (atomic_save, build_tokenizer,
                                                    build_model, append_metric, wsd_lambda,
                                                    lr_lambda)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--data-name", default="ultrafineweb_en")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--d-embed", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-k", type=int, default=64)
    p.add_argument("--d-v", type=int, default=64)
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--decay-frac", type=float, default=0.30,
                   help="wsd only. 0 = warmup then flat forever, which is what you want "
                        "when a later phase does the annealing")
    p.add_argument("--schedule", choices=["wsd", "cosine"], default="wsd")
    p.add_argument("--max-steps", type=int, default=5185)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--mlm-prob", type=float, default=0.15)
    p.add_argument("--mask-span-min", type=int, default=2)
    p.add_argument("--mask-span-dist", choices=["uniform", "geometric"], default="uniform",
                   help="geometric reproduces SpanBERT's sampler (mean ~3.8 with a long "
                        "tail); uniform never produces a genuinely hard long span")
    p.add_argument("--mask-geom-p", type=float, default=0.2)
    p.add_argument("--mask-span-max", type=int, default=4)
    p.add_argument("--top-k-layers", type=int, default=3,
                   help="average this many top teacher blocks as the target")
    # data2vec's NLP recipe ramps 0.999 -> 0.9999, but over ~100k updates, where the
    # final averaging window (1/(1-tau) = 10,000 steps) is ~10% of training. On a
    # 15k-step run that window is 65% of the whole run and the teacher goes nearly
    # static -- measured: the student converges onto a ~1,300-step-stale target and
    # oscillates around it. Scaled down to keep the window ~1,000 steps. BYOL reports
    # the whole 0.9-0.999 band as workable, so this stays well inside safe territory,
    # and the predictor provides collapse resistance independently of the EMA.
    p.add_argument("--ema-start", type=float, default=0.996)
    p.add_argument("--ema-end", type=float, default=0.999,
                   help="momentum ramps start->end; higher = more stable but staler targets")
    p.add_argument("--init-from", default=None, help="warm-start BOTH student and teacher")
    p.add_argument("--resume", action="store_true",
                   help="continue from <ckpt-dir>/last.pt keeping weights, teacher and step. "
                        "Lets you change micro-batch / workers / compile mid-run without "
                        "throwing away progress.")
    p.add_argument("--num-workers", type=int, default=0,
                   help=">0 loads batches on worker threads; the GPU otherwise waits on "
                        "the main thread between steps (~10%% idle measured)")
    p.add_argument("--compile", action="store_true", help="torch.compile the encoders")
    p.add_argument("--ckpt-dir", default="checkpoints/ckpt_jepa")
    p.add_argument("--ckpt-every", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokenizer", default="allenai/OLMo-1B-hf")
    p.add_argument("--limit-tokens", type=int, default=None)
    return p.parse_args()


class Predictor(nn.Module):
    """Narrow head mapping student hidden states into the teacher's target space.

    Kept deliberately small: if the predictor is powerful it can absorb the task and
    let the encoder off the hook, which is the whole point of I-JEPA's narrow predictor.
    """

    def __init__(self, d_model, hidden=None):
        super().__init__()
        h = hidden or d_model
        self.net = nn.Sequential(nn.Linear(d_model, h), nn.GELU(), nn.LayerNorm(h),
                                 nn.Linear(h, d_model))

    def forward(self, x):
        return self.net(x)


def build_targets(hidden_states, top_k):
    """Average the top-K teacher blocks after normalising each over the feature dim.

    Per-layer normalisation matters: without it the layer with the largest activation
    scale dominates the average and the target degenerates to that one layer.
    """
    layers = hidden_states[-top_k:]
    normed = [F.layer_norm(h, (h.size(-1),)) for h in layers]
    return torch.stack(normed, 0).mean(0)


@torch.no_grad()
def collapse_stats(target, pred):
    """Cheap collapse telemetry.

    - std: per-dimension spread across positions. -> 0 means every position emits the
      same vector, the degenerate solution.
    - eff_rank: exp(entropy of the normalised singular-value spectrum). A collapsed
      representation occupies ~1 direction; a healthy one occupies many.
    - cos_to_mean: how close each vector is to the batch mean. -> 1 means collapsed.
    """
    t = target.reshape(-1, target.size(-1)).float()
    std = t.std(0).mean().item()
    tc = t - t.mean(0, keepdim=True)
    try:
        sv = torch.linalg.svdvals(tc[: min(2048, tc.size(0))])
        p = sv / sv.sum().clamp_min(1e-9)
        eff_rank = float(torch.exp(-(p * p.clamp_min(1e-9).log()).sum()))
    except Exception:
        eff_rank = float("nan")
    cos_to_mean = F.cosine_similarity(t, t.mean(0, keepdim=True).expand_as(t), dim=-1).mean().item()
    p = pred.reshape(-1, pred.size(-1)).float()
    cos_raw = F.cosine_similarity(p, t, dim=-1).mean().item()
    # CENTRED cosine is the metric that means anything. Raw cosine on transformer
    # representations is ~0.9 for almost any pair because the space is anisotropic --
    # measured: a predictor emitting a CONSTANT scores 0.898 raw. Centring removes the
    # shared direction, so 0.0 is the true "learned nothing" baseline.
    pc = p - p.mean(0, keepdim=True)
    cos_centred = F.cosine_similarity(pc, tc, dim=-1).mean().item()
    return std, eff_rank, cos_to_mean, cos_centred, cos_raw


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    use_amp = device.type == "cuda"

    tk, ids = build_tokenizer(cfg.tokenizer)
    student = build_model(cfg, ids).to(device)
    resume_path = os.path.join(cfg.ckpt_dir, "last.pt")
    resuming = cfg.resume and os.path.exists(resume_path)
    ck = None
    if resuming:
        ck = torch.load(resume_path, map_location=device)
        student.load_state_dict(ck["model"])
        teacher = build_model(cfg, ids).to(device)
        teacher.load_state_dict(ck["teacher"])
        predictor = Predictor(cfg.d_model).to(device)
        predictor.load_state_dict(ck["predictor"])
        start_step = ck["step"]
        print(f"[resume] from step {start_step} in {resume_path}", flush=True)
    else:
        if not cfg.init_from:
            raise SystemExit("need --init-from (or --resume with an existing last.pt)")
        student.load_state_dict(torch.load(cfg.init_from, map_location=device)["model"])
        teacher = copy.deepcopy(student).to(device)
        predictor = Predictor(cfg.d_model).to(device)
        start_step = 0
        print(f"[init] student+teacher warm-started from {cfg.init_from}", flush=True)
    for p_ in teacher.parameters():
        p_.requires_grad_(False)
    teacher.eval()

    def build_ds(split):
        specs = []
        for part in [s.strip() for s in cfg.data_name.split(",") if s.strip()]:
            bits = part.split(":")
            specs.append((bits[0], float(bits[1]) if len(bits) > 1 and bits[1] else 1.0,
                          int(float(bits[2])) if len(bits) > 2 and bits[2] else cfg.limit_tokens))
        dss = [PackedMemmapDataset(cfg.data_dir, n, cfg.seq_len, split, limit_tokens=l)
               for n, _, l in specs]
        return dss[0] if len(dss) == 1 else MixtureDataset(dss, [w for _, w, _ in specs])

    train_ds, val_ds = build_ds("train"), build_ds("val")
    g = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.micro_batch, shuffle=True,
                              drop_last=True, generator=g, num_workers=cfg.num_workers,
                              persistent_workers=cfg.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.micro_batch, shuffle=False)

    params = list(student.parameters()) + list(predictor.parameters())
    decay = [p_ for p_ in params if p_.requires_grad and p_.ndim >= 2]
    nodecay = [p_ for p_ in params if p_.requires_grad and p_.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=cfg.lr, betas=(0.9, 0.98), eps=1e-8)
    if cfg.compile:
        # Student only. Compiling the TEACHER hits an inductor bug on this version:
        # torch.compile + no_grad + output_hidden_states raises
        #   AttributeError: 'float' object has no attribute 'meta'
        # (verified by bisection -- every submodule, the full encoder, autocast and
        # forward+backward all compile individually; only that combination fails).
        # Little is lost: the student does forward AND backward, ~75% of the compute,
        # while the teacher is a no_grad forward.
        student.encoder = torch.compile(student.encoder)
        print("[compile] student encoder compiled; teacher left eager "
              "(inductor bug with no_grad + hidden states)", flush=True)

    warm = int(cfg.warmup_frac * cfg.max_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, (lambda s: wsd_lambda(s, warm, cfg.max_steps, cfg.decay_frac))
        if cfg.schedule == "wsd" else
        (lambda s: lr_lambda(s, warm, cfg.max_steps)))

    if resuming:
        if ck.get("opt"):
            opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
            print("[resume] optimiser + schedule restored", flush=True)
        else:
            # Older checkpoints predate optimiser saving: rebuild the LR position from
            # the step count. Adam moments are lost, costing a brief transient.
            for _ in range(start_step):
                sched.step()
            print(f"[resume] no optimiser state in checkpoint -- schedule fast-forwarded to "
                  f"step {start_step}, Adam moments restart", flush=True)

    metrics_path = os.path.join(cfg.ckpt_dir, "metrics.jsonl")
    n_params = sum(p_.numel() for p_ in student.parameters())
    append_metric(metrics_path, {"t": "meta", "max_steps": cfg.max_steps, "start_step": 0,
                                 "params_M": round(n_params / 1e6, 1),
                                 "eff_batch": cfg.micro_batch * cfg.grad_accum,
                                 "tokens_per_step": cfg.micro_batch * cfg.grad_accum * cfg.seq_len,
                                 "vocab": ids["vocab_size"], "dataset": cfg.data_name,
                                 "lr": cfg.lr, "objective": "jepa-latent",
                                 "schedule": cfg.schedule, "seq_len": cfg.seq_len,
                                 "span": [cfg.mask_span_min, cfg.mask_span_max],
                                 "time": time.time()})

    def masked_forward(block):
        """Student sees [MASK] spans; teacher sees the original. mask_prob=1.0 because
        data2vec replaces every selected position -- the 80/10/10 split exists to stop
        the model keying on the [MASK] symbol when predicting *tokens*, which is not
        what is being predicted here."""
        masked, labels = mask_tokens(block, ids["mask_id"], ids["vocab_size"],
                                     special_token_ids=ids["special_ids"],
                                     pad_token_id=ids["pad_id"], mlm_probability=cfg.mlm_prob,
                                     mask_prob=1.0, random_prob=0.0,
                                     span_min=cfg.mask_span_min, span_max=cfg.mask_span_max,
                                     span_dist=cfg.mask_span_dist, geom_p=cfg.mask_geom_p)
        sel = labels != IGNORE_INDEX
        return masked.to(device), sel.to(device)

    def step_loss(block):
        masked, sel = masked_forward(block)
        orig = block.to(device)
        with torch.no_grad():
            _, t_hidden = teacher.encoder(orig, output_hidden_states=True)
            target = build_targets(t_hidden, cfg.top_k_layers)
        s_out = student.encoder(masked)
        pred = predictor(s_out)
        if sel.sum() == 0:
            return None, None, None
        return F.smooth_l1_loss(pred[sel], target[sel]), pred[sel].detach(), target[sel]

    @torch.no_grad()
    def evaluate(max_batches=20):
        student.eval(); predictor.eval()
        tot, n = 0.0, 0
        stats = None
        for i, block in enumerate(val_loader):
            if i >= max_batches:
                break
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss, pred, target = step_loss(block)
            if loss is None:
                continue
            tot += loss.item(); n += 1
            if stats is None:
                stats = collapse_stats(target, pred)
        student.train(); predictor.train()
        return tot / max(1, n), stats

    student.train(); predictor.train()
    step, micro, running = start_step, 0, 0.0
    data_iter = iter(train_loader)
    opt.zero_grad(set_to_none=True)
    t0 = time.time()

    while step < cfg.max_steps:
        try:
            block = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            block = next(data_iter)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss, _, _ = step_loss(block)
        if loss is None:
            continue
        (loss / cfg.grad_accum).backward()
        running += loss.item()
        micro += 1
        if micro == cfg.grad_accum:
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1; micro = 0

            # EMA teacher update, momentum ramping start -> end
            m = cfg.ema_start + (cfg.ema_end - cfg.ema_start) * (step / cfg.max_steps)
            with torch.no_grad():
                for pt, ps in zip(teacher.parameters(), student.parameters()):
                    pt.mul_(m).add_(ps.detach(), alpha=1 - m)
                for bt, bs in zip(teacher.buffers(), student.buffers()):
                    bt.copy_(bs)

            if step % cfg.log_every == 0:
                avg = running / (cfg.log_every * cfg.grad_accum)
                lr_now = sched.get_last_lr()[0]
                print(f"step {step:6d}/{cfg.max_steps} | loss {avg:6.4f} | lr {lr_now:.2e} | m {m:.5f}",
                      flush=True)
                append_metric(metrics_path, {"t": "train", "step": step, "loss": round(avg, 4),
                                             "lr": lr_now, "ema": round(m, 5), "time": time.time()})
                running = 0.0

            if step % cfg.eval_every == 0:
                vl, st = evaluate()
                std, rank, cos_mean, cos_pred, cos_raw = st if st else (float("nan"),) * 5
                # masked_acc slot carries cosine(pred, target) so the dashboard charts
                # something meaningful in [0,1]; there is no token accuracy here.
                append_metric(metrics_path, {"t": "eval", "step": step, "val_loss": round(vl, 4),
                                             "masked_acc": round(max(0.0, cos_pred), 4),
                                             "cos_raw": round(cos_raw, 4),
                                             "target_std": round(std, 4),
                                             "eff_rank": round(rank, 2),
                                             "cos_to_mean": round(cos_mean, 4),
                                             "time": time.time()})
                flag = "  <-- COLLAPSING" if (std < 0.05 or cos_mean > 0.95) else ""
                print(f"  eval {step}: loss {vl:.4f} | cos_centred {cos_pred:.4f} (raw {cos_raw:.3f}) | "
                      f"target_std {std:.4f} eff_rank {rank:.1f} cos_to_mean {cos_mean:.4f}{flag}",
                      flush=True)

            if step % cfg.ckpt_every == 0 or step == cfg.max_steps:
                atomic_save({"model": student.state_dict(), "predictor": predictor.state_dict(),
                             "teacher": teacher.state_dict(), "step": step, "cfg": vars(cfg),
                             # saved so a later resume is lossless, unlike this one
                             "opt": opt.state_dict(), "sched": sched.state_dict()},
                            os.path.join(cfg.ckpt_dir, "last.pt"))

    print(f"[done] {cfg.max_steps} steps in {(time.time()-t0)/60:.1f} min -> {cfg.ckpt_dir}/last.pt",
          flush=True)


if __name__ == "__main__":
    main()
