"""MLM pretraining for the v2 architecture: 6 layers, d_ff 1792, GELU.

v2 exists because three things were measured on ckpt3:

  * d_ff 3072 was ~2.4x oversized -- 99% of activation variance fitted in <=1228
    directions, so the surplus funds depth instead.
  * layer 0 had 2,755 of 3,072 ReLU units permanently dead (a dying-ReLU signature),
    leaving ~317 doing the work. GELU cannot die the same way.
  * residual-stream rank was still climbing at the last layer (483 -> 523 of 768),
    so more layers still have rank to build, whereas wider FFN or attention would add
    capacity the signal cannot fill.

Same CLI, metrics format and resume semantics as pretrain.py, so the dashboard and
every downstream script work unchanged. Start it from tools/build_v2_from_ckpt3.py's
output to inherit ckpt3's encoder rather than training from scratch.

    python -m mini_enc_transformer.training.pretrain_v2 \
        --init-from checkpoints/ckpt_v2_init.pt --ckpt-dir checkpoints/ckpt_v2 ...
"""
import argparse
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from mini_enc_transformer.data.dataset import MixtureDataset, PackedMemmapDataset
from mini_enc_transformer.model_v2.mlm import BertForMaskedLM, mask_tokens, IGNORE_INDEX
from mini_enc_transformer.training.pretrain import (append_metric, atomic_save,
                                                    build_tokenizer, lr_lambda,
                                                    make_optimizer, wsd_lambda)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--data-name", default="ultrafineweb_en")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--limit-tokens", type=int, default=None)
    # --- v2 architecture defaults ---
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--d-embed", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--d-k", type=int, default=64)
    p.add_argument("--d-v", type=int, default=64)
    p.add_argument("--d-ff", type=int, default=1792)
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    # The transfer is disrupted (FFN pruned 3072->1792, ReLU->GELU on ReLU-trained
    # weights, layers 4-5 duplicated from 2-3), measuring mlm_loss 8.05 vs ckpt3's 2.37,
    # which argues for a long ramp. But the new layers are STACKED COPIES of trained
    # layers, not random init, so there is no noise to protect the inherited weights
    # from -- the perturbation is structural and gets fixed by training, not by waiting.
    p.add_argument("--warmup-frac", type=float, default=0.10)
    p.add_argument("--schedule", choices=["cosine", "wsd"], default="wsd")
    p.add_argument("--decay-frac", type=float, default=0.30)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--mlm-prob", type=float, default=0.15)
    p.add_argument("--mask-span-dist", choices=["uniform", "geometric"], default="geometric")
    p.add_argument("--mask-geom-p", type=float, default=0.2)
    p.add_argument("--mask-span-min", type=int, default=1)
    p.add_argument("--mask-span-max", type=int, default=10)
    p.add_argument("--eval-span-min", type=int, default=1,
                   help="eval masking; defaults to scattered so masked_acc stays "
                        "comparable with every earlier run")
    p.add_argument("--eval-span-max", type=int, default=1)
    p.add_argument("--ckpt-dir", default="checkpoints/ckpt_v2")
    p.add_argument("--ckpt-every", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--init-from", default=None, help="weights only; fresh optimiser/schedule")
    p.add_argument("--resume", action="store_true")
    # --- grafted-layer support (see tools/graft_l4.py) -------------------------
    p.add_argument("--boost-layers", default="",
                   help="comma-separated layer indices to put in their own param group: "
                        "higher LR and NO weight decay. A freshly grafted layer and a "
                        "converged stack need different hyperparameters -- uniform ones "
                        "are what let the first stacked graft die.")
    p.add_argument("--boost-mult", type=float, default=5.0,
                   help="LR multiplier for --boost-layers, annealed back to 1.0")
    p.add_argument("--boost-hold-frac", type=float, default=0.15,
                   help="fraction of REMAINING steps to hold the multiplier before "
                        "cosine-annealing it to 1.0 over the rest of the run")
    p.add_argument("--boost-wd", type=float, default=0.0,
                   help="weight decay for the boosted group. Zero by default: these "
                        "matrices start at zero and receive only weak gradient until "
                        "they lift off it, so decay competes with the very signal that "
                        "is supposed to grow them.")
    p.add_argument("--rewarm-steps", type=int, default=0,
                   help="linear LR ramp over this many steps from the resume point, on "
                        "top of the base schedule. Grafting changes the loss surface; "
                        "re-entering at full LR spikes it.")
    p.add_argument("--reset-opt-layers", default="",
                   help="drop Adam moments for these layers on resume. Required after a "
                        "graft: the saved moments describe weights that no longer exist.")
    p.add_argument("--freeze-except-layers", default="",
                   help="train ONLY these encoder layers; freeze every other parameter "
                        "including embeddings, the MLM head and the other blocks. The "
                        "point is to remove the escape route: with the rest of the stack "
                        "frozen, the loss can only improve through these layers, so they "
                        "cannot be routed around the way a jointly-trained layer can.")
    p.add_argument("--probe-layers", default="4,5",
                   help="layers to report head entropy and FFN never-positive for, "
                        "written into metrics.jsonl at every eval")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokenizer", default="allenai/OLMo-1B-hf")
    return p.parse_args()


def build_model_v2(cfg, ids):
    return BertForMaskedLM(vocab_size=ids["vocab_size"], d_model=cfg.d_model, d_k=cfg.d_k,
                           d_v=cfg.d_v, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
                           d_ff=cfg.d_ff, pad_id=ids["pad_id"], d_embed=cfg.d_embed)


@torch.no_grad()
def evaluate(model, val_loader, ids, device, max_batches=20, span_min=1, span_max=1, seed=1234):
    """Fixed masking seed so every eval scores the same corrupted positions."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    tot_loss, correct, scored, n = 0.0, 0, 0, 0
    for block in val_loader:
        if n >= max_batches:
            break
        n += 1
        masked, labels = mask_tokens(block, ids["mask_id"], ids["vocab_size"],
                                     special_token_ids=ids["special_ids"],
                                     pad_token_id=ids["pad_id"], generator=g,
                                     span_min=span_min, span_max=span_max)
        masked, labels = masked.to(device), labels.to(device)
        out = model(masked, labels)
        tot_loss += out["loss"].item()
        sel = labels != IGNORE_INDEX
        correct += (out["logits"].argmax(-1)[sel] == labels[sel]).sum().item()
        scored += sel.sum().item()
    model.train()
    return tot_loss / max(1, n), correct / max(1, scored)


def parse_layers(s):
    return [int(x) for x in str(s).split(",") if x.strip()]


def make_optimizer_boost(model, lr, wd, boost_layers, boost_wd):
    """Base param groups, plus separate groups for grafted layers.

    A freshly grafted layer and a converged stack want different hyperparameters: the
    graft needs a high LR to move at all, and zero weight decay because its output
    projections start at zero and decay would fight the weak early gradient that is
    meant to grow them.

    Returns (optimizer, boost_flags); boost_flags[i] marks param group i as boosted, so
    the LambdaLR can be given a different lambda per group.

    With boost_layers empty this produces exactly the two groups make_optimizer does,
    in the same order, so an optimiser state saved by an earlier run still loads.
    """
    pref = tuple(f"encoder.encoder_blocks.{i}." for i in boost_layers)
    buckets = {("base", True): [], ("base", False): [],
               ("boost", True): [], ("boost", False): []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        which = "boost" if (pref and n.startswith(pref)) else "base"
        buckets[(which, p.ndim >= 2)].append(p)

    groups, flags = [], []
    for (which, is_2d), ps in buckets.items():
        if not ps:
            continue
        # 1D params (biases, LayerNorm gains) never decay, in either group
        groups.append({"params": ps,
                       "weight_decay": (wd if which == "base" else boost_wd) if is_2d else 0.0})
        flags.append(which == "boost")
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.98), eps=1e-6), flags


def make_lambdas(cfg, warm, boost_flags, resume_at):
    """Base schedule with two overlays: a linear re-warm from the resume point, and a
    multiplier on the grafted layers that holds, then cosine-anneals back to 1.0."""
    total = cfg.max_steps
    remaining = max(1, total - resume_at)
    hold_end = resume_at + int(cfg.boost_hold_frac * remaining)

    def base(s):
        v = (lr_lambda(s, warm, total) if cfg.schedule == "cosine"
             else wsd_lambda(s, warm, total, cfg.decay_frac))
        if cfg.rewarm_steps > 0:
            v *= min(1.0, max(0.0, s - resume_at + 1) / cfg.rewarm_steps)
        return v

    def boost(s):
        if s <= hold_end:
            m = cfg.boost_mult
        else:
            prog = min(1.0, (s - hold_end) / max(1, total - hold_end))
            m = 1.0 + (cfg.boost_mult - 1.0) * 0.5 * (1.0 + math.cos(math.pi * prog))
        return base(s) * m

    return [boost if f else base for f in boost_flags]


@torch.no_grad()
def layer_probe(model, batch, layers):
    """Attention entropy and FFN never-positive fraction for the named layers.

    These are the two numbers that diagnosed the dead graft in the first place: entropy
    near log(T) means attention is averaging rather than selecting, and never_pos near 1
    means the FFN is switched off. Logging them per eval makes the recovery visible
    while it happens instead of only in a post-mortem.
    """
    if not layers:
        return {}
    was_training = model.training
    model.eval()
    B, T = batch.shape
    out = {}
    x = model.encoder.pe(model.encoder.embedding(batch))
    for i, blk in enumerate(model.encoder.encoder_blocks):
        m, ff = blk.mha, blk.feed_forward
        if i in layers:
            # recomputed explicitly: the fused SDPA path never materialises the probs
            q = (x @ m.W_q).view(B, T, m.n_heads, m.d_k).transpose(1, 2)
            k = (x @ m.W_k).view(B, T, m.n_heads, m.d_k).transpose(1, 2)
            att = torch.softmax(q @ k.transpose(-2, -1) / m.d_k ** 0.5, dim=-1)
            out[f"l{i}_ent"] = round(
                float(-(att * att.clamp_min(1e-12).log()).sum(-1).mean()), 4)
            del att, q, k
        h = blk.layer_norm_1(m(x, False) + x)
        H = ff.act(h @ ff.l1 + ff.b1)
        if i in layers:
            out[f"l{i}_np"] = round(
                float((H.reshape(-1, H.shape[-1]).max(0).values <= 0).float().mean()), 4)
        x = blk.layer_norm_2(ff(h) + h)
    if was_training:
        model.train()
    return out


def fmt_probe(pm, layers):
    return " | ".join(f"L{i} ent {pm.get(f'l{i}_ent', float('nan')):.2f} "
                      f"np {100*pm.get(f'l{i}_np', float('nan')):.0f}%" for i in layers)


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    use_amp = device.type == "cuda"

    tk, ids = build_tokenizer(cfg.tokenizer)
    model = build_model_v2(cfg, ids).to(device)
    boost_layers = parse_layers(cfg.boost_layers)
    probe_layers = parse_layers(cfg.probe_layers)

    # Freeze BEFORE building the optimiser -- make_optimizer_boost filters on
    # requires_grad, so frozen tensors never enter a param group at all.
    freeze_except = parse_layers(cfg.freeze_except_layers)
    if freeze_except:
        keep = tuple(f"encoder.encoder_blocks.{i}." for i in freeze_except)
        tot = 0
        for n, prm in model.named_parameters():
            prm.requires_grad = n.startswith(keep)
            tot += prm.numel() if prm.requires_grad else 0
        allp = sum(prm.numel() for prm in model.parameters())
        print(f"[freeze] training ONLY layers {freeze_except}: "
              f"{tot/1e6:.3f}M of {allp/1e6:.3f}M params ({100*tot/allp:.1f}%)", flush=True)
    opt, boost_flags = make_optimizer_boost(model, cfg.lr, cfg.weight_decay,
                                            boost_layers, cfg.boost_wd)
    warm = int(cfg.warmup_frac * cfg.max_steps)

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
                              drop_last=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=cfg.micro_batch, shuffle=False)

    start_step = 0
    last_path = os.path.join(cfg.ckpt_dir, "last.pt")
    resuming = cfg.resume and os.path.exists(last_path)
    if cfg.init_from and not resuming:
        ck = torch.load(cfg.init_from, map_location=device)
        model.load_state_dict(ck["model"])
        src = ck.get("built_from", cfg.init_from)
        print(f"[init] v2 weights from {cfg.init_from} (built from {src})", flush=True)
    sched_state, graft_record = None, None
    if resuming:
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"]); start_step = ck["step"]
        # Optimiser state is keyed by position within the group layout, so it only loads
        # into an optimiser shaped the same way. --boost-layers changes that shape, so
        # rebuild the ORIGINAL layout, load into that, and carry the moments across by
        # parameter identity -- the tensors are the same objects either way.
        try:
            opt.load_state_dict(ck["opt"])
        except ValueError:
            # The saved state was written under a different --boost-layers, so it has a
            # different group count. Rebuild THAT layout (the checkpoint records it in
            # cfg), load into it, and carry the moments across by parameter identity.
            old_boost = parse_layers(ck.get("cfg", {}).get("boost_layers", "") or "")
            tmp, _ = make_optimizer_boost(model, cfg.lr, cfg.weight_decay,
                                          old_boost, cfg.boost_wd)
            try:
                tmp.load_state_dict(ck["opt"])
                opt.state.update(dict(tmp.state))
                print(f"[resume] carried Adam moments for {len(tmp.state)} tensors across "
                      f"a param-group change (was boost={old_boost}, now "
                      f"{boost_layers})", flush=True)
            except ValueError:
                print("[resume] optimiser layout could not be matched; continuing with "
                      "FRESH Adam moments for all params", flush=True)
        sched_state = ck.get("sched")
        torch.set_rng_state(ck["rng"].cpu().to(torch.uint8))
        if use_amp and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu().to(torch.uint8))
        if ck.get("graft"):
            graft_record = ck["graft"]
            print(f"[resume] checkpoint carries a graft: {graft_record}", flush=True)
        print(f"[resume] from step {start_step}", flush=True)

    # A grafted layer's saved moments describe weights that no longer exist. Adam would
    # otherwise divide the new gradients by the old second moment for several hundred
    # steps, which is exactly when the layer needs to move fastest.
    reset_layers = parse_layers(cfg.reset_opt_layers)
    if reset_layers:
        pref = tuple(f"encoder.encoder_blocks.{i}." for i in reset_layers)
        cleared = [n for n, p in model.named_parameters()
                   if n.startswith(pref) and opt.state.pop(p, None) is not None]
        print(f"[resume] cleared Adam moments for {len(cleared)} tensors in layers "
              f"{reset_layers}", flush=True)

    lambdas = make_lambdas(cfg, warm, boost_flags, start_step)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambdas)
    if resuming:
        plain = (sched_state is not None and not cfg.rewarm_steps and not boost_layers
                 and len(sched_state.get("_last_lr", [])) == len(opt.param_groups))
        if plain:
            sched.load_state_dict(sched_state)
        else:
            # re-seed rather than restore: the lambdas are pure functions of absolute
            # step, so the schedule is fully determined by last_epoch
            sched.last_epoch = start_step
            for g, lam in zip(opt.param_groups, sched.lr_lambdas):
                g["lr"] = g["initial_lr"] * lam(start_step)
            sched._last_lr = [g["lr"] for g in opt.param_groups]
            print(f"[resume] schedule re-seeded at step {start_step}: rewarm "
                  f"{cfg.rewarm_steps} steps, boost x{cfg.boost_mult} (wd {cfg.boost_wd}) "
                  f"on layers {boost_layers}", flush=True)
            print(f"[resume] group LRs now: "
                  + ", ".join(f"{'boost' if f else 'base'} {g['lr']:.2e}"
                              for g, f in zip(opt.param_groups, boost_flags)), flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] v2 {n_params/1e6:.1f}M | {cfg.n_layers} layers | d_ff {cfg.d_ff} | GELU "
          f"| device {device} | eff batch {cfg.micro_batch*cfg.grad_accum}", flush=True)

    metrics_path = os.path.join(cfg.ckpt_dir, "metrics.jsonl")
    tokens_per_step = cfg.micro_batch * cfg.grad_accum * cfg.seq_len
    append_metric(metrics_path, {"t": "meta", "max_steps": cfg.max_steps,
                                 "start_step": start_step, "params_M": round(n_params/1e6, 1),
                                 "eff_batch": cfg.micro_batch * cfg.grad_accum,
                                 "tokens_per_step": tokens_per_step, "vocab": ids["vocab_size"],
                                 "dataset": cfg.data_name, "lr": cfg.lr, "arch": "v2-gelu",
                                 "n_layers": cfg.n_layers, "d_ff": cfg.d_ff,
                                 "seq_len": cfg.seq_len, "schedule": cfg.schedule,
                                 "boost_layers": boost_layers, "boost_mult": cfg.boost_mult,
                                 "boost_wd": cfg.boost_wd, "rewarm_steps": cfg.rewarm_steps,
                                 "probe_layers": probe_layers,
                                 "freeze_except": freeze_except,
                                 "time": time.time()})

    # Baseline before any training: the transfer (prune + GELU + stacking) changed the
    # function, and knowing the starting damage is how we judge recovery.
    # Fixed probe batch: same sequences at every eval, so the layer traces are
    # comparable step to step rather than reflecting which batch happened to come up.
    probe_batch = next(iter(val_loader))[:8].to(device) if probe_layers else None

    vl, va = evaluate(model, val_loader, ids, device,
                      span_min=cfg.eval_span_min, span_max=cfg.eval_span_max)
    pm = layer_probe(model, probe_batch, probe_layers)
    print(f"[init] pre-training eval: mlm_loss {vl:.4f} masked_acc {va:.4f}"
          + (f" | {fmt_probe(pm, probe_layers)}" if pm else ""), flush=True)
    append_metric(metrics_path, {"t": "eval", "step": start_step, "val_loss": round(vl, 4),
                                 "masked_acc": round(va, 4), **pm, "time": time.time()})

    model.train()
    step, micro, running = start_step, 0, 0.0
    data_iter = iter(train_loader)
    opt.zero_grad(set_to_none=True)
    t0 = time.time()
    while step < cfg.max_steps:
        try:
            block = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader); block = next(data_iter)
        masked, labels = mask_tokens(block, ids["mask_id"], ids["vocab_size"],
                                     special_token_ids=ids["special_ids"],
                                     pad_token_id=ids["pad_id"], mlm_probability=cfg.mlm_prob,
                                     span_min=cfg.mask_span_min, span_max=cfg.mask_span_max,
                                     span_dist=cfg.mask_span_dist, geom_p=cfg.mask_geom_p)
        masked, labels = masked.to(device), labels.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss = model(masked, labels)["loss"]
        (loss / cfg.grad_accum).backward()
        running += loss.item(); micro += 1
        if micro == cfg.grad_accum:
            torch.nn.utils.clip_grad_norm_(
                [prm for prm in model.parameters() if prm.requires_grad], cfg.grad_clip)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1; micro = 0
            if step % cfg.log_every == 0:
                avg = running / (cfg.log_every * cfg.grad_accum)
                lr_now = sched.get_last_lr()[0]
                print(f"step {step:6d}/{cfg.max_steps} | loss {avg:6.3f} | lr {lr_now:.2e}",
                      flush=True)
                rec = {"t": "train", "step": step, "loss": round(avg, 4),
                       "lr": lr_now, "tokens": step * tokens_per_step, "time": time.time()}
                if boost_layers:
                    rec["lr_boost"] = next(g["lr"] for g, f in
                                           zip(opt.param_groups, boost_flags) if f)
                append_metric(metrics_path, rec)
                running = 0.0
            if step % cfg.eval_every == 0:
                vl, va = evaluate(model, val_loader, ids, device,
                                  span_min=cfg.eval_span_min, span_max=cfg.eval_span_max)
                pm = layer_probe(model, probe_batch, probe_layers)
                print(f"  eval {step}: val_loss {vl:.4f} masked_acc {va:.4f}"
                      + (f" | {fmt_probe(pm, probe_layers)}" if pm else ""), flush=True)
                append_metric(metrics_path, {"t": "eval", "step": step, "val_loss": round(vl, 4),
                                             "masked_acc": round(va, 4), **pm,
                                             "time": time.time()})
            if step % cfg.ckpt_every == 0 or step == cfg.max_steps:
                atomic_save({"model": model.state_dict(), "opt": opt.state_dict(),
                             "sched": sched.state_dict(), "step": step,
                             "rng": torch.get_rng_state(),
                             "cuda_rng": torch.cuda.get_rng_state() if use_amp else None,
                             # carry provenance forward: atomic_save builds a fresh dict,
                             # so without this the graft record survives only until the
                             # first checkpoint write after the resume
                             "graft": graft_record,
                             "cfg": vars(cfg)}, last_path)
    print(f"[done] trained to step {step} in {(time.time()-t0)/60:.1f} min -> {last_path}",
          flush=True)


if __name__ == "__main__":
    main()
