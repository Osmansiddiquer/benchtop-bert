# BERT-from-scratch — Work Log

> What I did while you were asleep. **TL;DR:** the model is re-architected
> (factorized embedding), the full BERT-MLM pretraining pipeline is built and
> **every piece is verified end-to-end**, and it's wired into `BERT.ipynb`. The
> only thing not done is a *real* long training run — blocked purely by HF
> download speed (~0.26 MB/s unauthenticated). Set `HF_TOKEN` and you're go.

---

## ✅ PHASE 1 COMPLETE (WikiText-103)
Finished at step 8,800 in ~123 min compute. **Final val_loss 2.544, best masked_acc 55.7%.**
Checkpoint: `ckpt/last.pt` (this is the general backbone; a copy gets fine-tuned per
downstream task, e.g. SST-2). Trajectory matched the scaling-law read — accuracy
plateaued ~49% mid-run then climbed to ~55% through the cosine anneal.

**Bug fixed (data_prep.py):** UltraFineWeb's `score` field arrives as a **string**;
the `score < min_score` filter raised TypeError → infinite retry → 0 tokens for ~28 min.
Now coerces `float(score)`. UFW download restarted clean and is packing tokens.

## 🌙 UNATTENDED PLAN (user away ~6h from 2026-08-04 ~04:15)
- **Phase-2 running**: continued pretraining on ~340M UltraFineWeb tokens, budget **2 epochs
  = 20,789 steps** (`ckpt2/`), warm-started from `ckpt/last.pt`. Healthy (loss std ~0.02,
  masked_acc ~46% and rising, no spikes/NaN).
- **Monitor** checks every ~18 min: stability watch (spike/NaN/rising-val-loss), auto-resume
  across laptop sleeps, keeps the dashboard alive.
- **On phase-2 completion** the monitor runs `finetune_sst2.py` → full fine-tune on SST-2
  (mean-pool + linear head, val accuracy) → writes `finetune_sst2_results.json`, then stops.
  So on return there should be a real **SST-2 validation accuracy** to look at.
- ~17k steps remain; at ~1.1s/step that's ~5h of *compute*, but the laptop sleeps, so it may
  or may not finish within 6h. Whatever's reached is checkpointed; nothing is lost.
- **Decision left to you:** whether to pretrain *further* (I'll report the val-loss slope as the
  signal) — I will NOT auto-extend pretraining beyond the 2 epochs.

## 🔁 PHASE 2 — continued pretraining on UltraFineWeb (staged, auto-starts)
Per the scaling-law read (we're data-limited, not compute-limited), the next
optimization for the laptop encoder is *better data*, not more WikiText steps.
- **UltraFineWeb download is running now** (`data/ultrafineweb_en.*`, authenticated,
  resumable, direct-shard). It accumulates at ~0.35 MB/s.
- **When phase-1 finishes AND ≥80M UFW tokens are packed**, the monitor runs
  `launch_phase2.sh`: warm-starts from `ckpt/last.pt` (`train.py --init-from`), fresh
  cosine schedule at **lr 5e-4** (gentler, to adapt without wrecking features), writes
  to **`ckpt2/`** (separate metrics so dashboard curves don't collide), ~2 epochs of
  whatever UFW is ready (capped 8000 steps). Dashboard auto-repoints to `ckpt2/metrics.jsonl`.
- Phase-2 is warm-start (weights only, fresh optimizer/step) — verified via `--init-from`.
- Manual start any time: `bash launch_phase2.sh` (idempotent: waits for data, starts, or resumes).

## 📊 LIVE DASHBOARD → http://localhost:8000
Structured metrics now stream to `ckpt/metrics.jsonl` (one JSON line per log/eval).
`serve_dashboard.py` (stdlib only, no deps) serves a self-refreshing page — loss +
masked-accuracy curves, step/ETA/tokens tiles, LIVE/PAUSED/DONE status — polling every 3s.
```bash
python serve_dashboard.py --port 8000        # then open http://localhost:8000
```
Full history was backfilled from the old text log, so the curves start at step 0.

## 👉 When you wake up — do this
1. **Set an HF token** for fast downloads (the one real blocker):
   ```bash
   export HF_TOKEN=hf_...        # then relaunch prep
   ```
2. **Check prep progress** (I launched a resumable run at ~0.26 MB/s):
   ```bash
   cat data/ultrafineweb_en.manifest.json          # tokens_written so far
   ```
   If you set HF_TOKEN, kill it and relaunch for real:
   ```bash
   python data_prep.py --target-tokens 1_500_000_000 --out data --ufw-shards 8 --min-score 0.9
   ```
   (Resumable + survives sleep/12h outages — verified. Rerun the same command to continue.)
3. **Train**: open `BERT.ipynb`, run the cells top-to-bottom (or `python train.py --resume`).
   Defaults are tuned to your 4GB GPU (micro-batch 16 → ~1.8 GB peak).

---

## ⚠️ Sleep incident + fixes (read this)
- **The laptop slept ~14h during the first run.** My `--max-seconds 10800` used
  wall-clock (`time.time()`), which counts sleep, so it tripped the "3h" budget at
  **step 1851/8800** after only ~37 min of *actual* compute. Lesson: budget by STEPS,
  not wall-clock, on a machine that sleeps.
- **Fixed:** resumed with **no `--max-seconds`** — budget is now `max_steps 8800`
  (~3h of real compute). Sleep now only *pauses* the run; it accumulates steps whenever awake.
- **Bug found + fixed:** GPU resume crashed — `torch.load(map_location='cuda')` moved
  the saved RNG state to GPU, but `set_rng_state` needs a CPU ByteTensor. Now coerced
  back to CPU (train.py). (My earlier resume test passed only because it ran on CPU.)
- **Status:** resumed cleanly `[resume] from step 1851`, now training to 8800.

## 🏁 END-TO-END RESULT (2026-08-04) — pretrain → SST-2

**SST-2 val accuracy: ~83.5% ± 1.3%** (per epoch: 82.11 → 84.63 → 84.75).
Backbone `ckpt2/last.pt`, mean-pooled, full fine-tune, lr 2e-5, batch 32, 3 epochs.

> ⚠️ **Quote ~83.5%, not 84.75%.** `best_val_acc` is the *maximum over 34 evals* on an
> 872-example dev set. At ±1.3% per-eval noise, a max over 34 draws is biased upward by
> ~1–2 SE, and 84.75% happened to land on the final eval. The mean of the epoch-3 evals
> (~83.5%) is the defensible estimate.

**The fine-tune overfit** (user spotted this; confirmed):

| | train | val_loss | val_acc |
|---|---|---|---|
| step 2105 (end ep1) | 0.278 | **0.3910** ← min | 0.8211 |
| step 4800 | 0.145 | 0.5138 | 0.8280 |
| step 6315 (end ep3) | 0.171 | 0.4431 | 0.8475 |

val_loss bottomed at the end of epoch 1 and rose thereafter while train loss kept
falling; the train/val gap went 1.4× → 3.1×.

**But do NOT early-stop on val_loss here.** Accuracy kept *improving* the whole time —
the classic overconfidence decoupling, where CE penalises growing certainty on the
errors while argmax decisions still improve. Stopping at the val_loss minimum would
have given 82.1%, ~2.5 points *worse*. If we want to attack it: layer-wise LR decay, or
dropout inside the encoder (there is none — the 0.1 is on the classifier head only).

This is a *fine-tuning* phenomenon on 67K labelled examples and is unrelated to the
pretraining capacity-bound argument below (which rests on the MLM val_loss plateau).

| | ours | BERT-base |
|---|---|---|
| params | 28.7M | 110M |
| pretrain tokens | 800M (118M WikiText + 681M UFW) | ~3.3B unique × 40 epochs |
| SST-2 dev | **~83.5%** | ~92.7% |

**We did not beat BERT — we're ~9 points short.** SST-2 dev is only 872 examples,
so 1 SE ≈ ±1.3%; a 9-point gap is ~7 SE and unambiguously real, not noise.

Sober framing: ~84.7% is roughly where strong *classical* baselines sit on SST-2
(NB-SVM with bigrams is low-80s; small CNNs mid-80s). So the pipeline is correct
end-to-end, but 28.7M params at 800M tokens buys about what a good non-pretrained
baseline gets. (Reference numbers from memory — verify before quoting anywhere.)

**Why, with evidence:** phase-2 val_loss plateaued at 1.9 epochs with per-window
improvement (0.024) *below* eval noise (0.047), while train and val loss stayed level
with each other and zero dropout produced no overfitting. All three point the same
way — **capacity-bound, not data-bound**. More epochs or more tokens at this size
will not close an 8-point gap; parameters will. The obvious next experiment is
~110M params (BERT-base size) rather than more UltraFineWeb.

## 🔌 AUTO-RECOVERY FROM SLEEP (2026-08-04, second sleep)
The laptop slept again and killed everything at step 5800. Nothing restarted it,
because every watchdog so far lived *inside* the session that the sleep suspended —
it went down at the same instant as the training it was meant to be watching, so it
could never observe the failure it existed to catch. Moved recovery out to the OS.

- **`supervisor.sh`** — argument-free state machine, safe to run at any moment
  (`flock` serialises ticks, every branch no-ops when the work is already running):
  results exist → done · train/finetune running → nothing · ckpt step ≥ max_steps →
  launch SST-2 fine-tune · else → resume phase-2. Also restarts the dashboard, which
  dies to sleep the same way.
- **crontab** — `*/2 * * * *` + `@reboot`. Cron is restarted by the system on wake,
  so it sits outside the blast radius. Edit with `crontab -e`; logs in `logs/`.
- **Two bugs that would have made this silently useless** (both fail *quietly* — the
  supervisor would appear installed and simply never start anything):
  1. `launch_phase2.sh` logged to `$CLAUDE_JOB_DIR/tmp`, which is unset under cron →
     redirect into a nonexistent dir → `set -e` aborts before launching. Now `logs/`.
  2. Bare `python` under cron's minimal PATH is `/usr/bin/python`, not conda's.
     PATH is now pinned in the crontab.
- **Verified by actually breaking it**, not by inspection: `kill -9`'d training at
  step 6020 and left it alone; cron restored it at 13:42:09 from step 6000. Cost of a
  hard kill is ≤ the 200-step checkpoint interval (~25 s).
### ⚠️ What cron does NOT cover (measured, not assumed)
The journal settles it — during Windows sleep the WSL VM is not paused, it is
**destroyed**:

```
boot -1:  Fri 07-31 17:03      ->  Tue 08-04 04:53:48   <- run died here
boot  0:  Tue 08-04 13:32:12   ->  (opened VS Code)
```

Two distinct boot IDs with an 8h38m hole. Immediately before the cutoff,
`systemd-resolved` logs "Clock change detected" every 30 s — the VM being frozen and
thawed as Windows went down — then termination. WSL2 tears the VM down seconds after
its last client disconnects, which is why VS Code closing is part of the causal chain,
not a coincidence. (`uptime -s` reports 05:47 and is simply wrong here; the WSL uptime
counter does not survive this. `who -b` and `journalctl --list-boots` agree on 13:32.)

So, honestly scoped:
- **Cron DOES cover:** crashes, OOM kills, train.py exiting — anything where WSL stays up.
- **Cron does NOT cover:** Windows sleep. There is no kernel running to fire it. No
  Linux-side mechanism can, and `@reboot` only helps once something re-invokes WSL.

**Decision (user, 2026-08-04): no Windows-side changes.**

### 🪲 The flock bug — and why the recovery test could never have caught it
`supervisor.sh` held its lock on fd 9. `setsid` children **inherit open fds**, so the
long-lived dashboard the supervisor itself spawned captured fd 9 and held the lock for
its whole lifetime. Every cron tick then died at `flock -n 9` and exited 0 — silently,
with nothing logged. The supervisor was inert for ~5 h; phase 2 finished on its own
momentum, unsupervised.

The 13:42 recovery test passed only because the dashboard alive at that moment had been
started from a shell, not by the supervisor — so it held no lock. **The act of the
supervisor launching a daemon is what disabled the supervisor**, which means that test
could not have detected the bug no matter how carefully it was run. Lesson: verifying a
watchdog once tells you nothing about whether it *stays* armed.

Fixed with `9>&-` on every spawn (+ a prominent comment, since a future edit could
silently reintroduce it), a `complete: true` done-check instead of file-existence, and a
3-attempt cap so a crashing fine-tune can't respawn on the GPU forever.

**Superseded 2026-08-04: cron removed entirely (`crontab -r`) at the user's request** —
they were staying awake at the keyboard, so the OS layer bought nothing and carried this
silent-failure class. `supervisor.sh` remains on disk but is unscheduled/inert; it is
still the right pattern for the H200 box, which has no WSL-teardown problem.

If that ever becomes annoying, the two Windows-side fixes (deliberately not applied)
are: a Task Scheduler task on wake/unlock running
`wsl.exe -d Ubuntu -u nimda .../supervisor.sh`, and/or `powercfg` keep-awake while a
run is active. Interop is available (`schtasks.exe`, `powershell.exe` are on PATH).

**Also verified:** train.py runs with PPID 1 and no controlling tty (fully detached via
`setsid`), so closing VS Code does not kill it while WSL is alive.

## 🚂 3-HOUR TRAINING RUN — IN PROGRESS
- **Dataset for this run: WikiText-103** (118.9M tokens, 929K blocks), NOT UltraFineWeb.
  **Why:** your HF link is a hard ~0.35 MB/s even with the token + hf_transfer (I tested
  — it's the network path, not a throttle). One UltraFineWeb shard = ~1h download, so a
  bigger chunk of it can't be ready for a run starting now. WikiText-103 fully downloads
  in ~6 min and is a solid clean-English pretraining base for a sentiment fine-tune.
  UltraFineWeb stays the target for a rerun on a faster link.
- **Budget:** `max_steps 8800` (cosine horizon = measured 3h of compute at 1.19 s/step).
  **Do NOT use `--max-seconds`** (it counts sleep — see incident above). Effective batch
  256, seq 128 → ~290M tokens seen (~2.4 epochs). Peak VRAM ~1.95 GB.
- **Config:** 768 d_model / E=128 / 4 heads / 4 layers, AdamW lr 1e-3, warmup 6% → cosine,
  bf16, grad-clip 1.0, MLM 15%.
- **Checkpoints:** `ckpt/last.pt` every 200 steps. If sleep kills it: `python train.py --resume`
  (add the same flags) continues toward step 8800. Progress log: `$CLAUDE_JOB_DIR/tmp/train.log`.
- **After it finishes:** `ckpt/last.pt` holds the pretrained encoder for your sentiment
  fine-tune. The notebook's last cell shows how to load it + probe masked tokens.

## Decisions locked (with you)
- **Factorized embedding, E=128** (ALBERT). Keeps OLMo tokenizer + d_model=768.
- **Dataset: UltraFineWeb `en`**, stream + `score>0.9` → local uint16 memmap → train offline.
- **4 heads, 4 blocks, seq 128, MLM only, no NSP.** OLMo tokenizer + added `[MASK]` (vocab 50281).

## Status board
| Step | What | State |
|------|------|-------|
| 0 | Encoder review + bug fixes (4 bugs) | ✅ DONE + verified |
| 0 | MLM module (mask / head / model) | ✅ DONE + verified |
| 1 | Factorized embedding (embeddings.py, encoder.py) | ✅ DONE + verified |
| 2 | ALBERT MLM head (mlm.py) | ✅ DONE + verified |
| 3 | data_prep.py (resumable stream→memmap) | ✅ DONE + verified |
| 4 | Training loop + checkpoint/resume (train.py, data.py) | ✅ DONE + verified |
| 5 | Wired into BERT.ipynb | ✅ DONE |
| — | **Real full training run** | ⏳ blocked on HF bandwidth (needs HF_TOKEN) |

## Verification results (all green)
- **Factorized model:** 28.7M params total; embedding **6.53M = 22.7%** (was ~58%);
  decoder tied to word_emb; init loss **10.86 ≈ ln(50281)**; grads reach word_emb + emb_proj.
- **Prep resume:** simulated a network drop mid-run → committed, backed off, resumed
  from the exact doc → **byte-identical** packed output, no corruption/dup; short +
  low-score docs filtered. (This is the 12h-outage robustness, proven.)
- **Training loop (CPU, learnable synthetic data):** loss **9.3 → 0.70**, masked
  accuracy **→ 1.00**; warmup→cosine LR, eval, atomic checkpointing all work.
- **Resume:** trained to step 60 (loss 2.18) → resumed → continued at 2.01 (no reset).
- **GPU (real hardware):** full 768/4/4 model trains under **bf16 on the 3050 Ti**;
  loss drops; checkpoint written.

## Key findings you should know
- **GPU fit (measured on your 3050 Ti, 4.29 GB total):**
  | micro-batch | peak VRAM | ms/step | verdict |
  |---|---|---|---|
  | 16 | 1.84 GB | 91 | ✅ safe default (leaves room for desktop) |
  | 32 | 3.25 GB | 188 | ✅ works if GPU otherwise idle |
  | 48 | 4.63 GB | 507 | ⚠️ spills to WSL2 shared mem, slow |
  | 64 | 6.03 GB | 1751 | ❌ heavy spill |
  Use micro-batch 16 (default) with grad-accum 16 → effective batch 256.
- **HF download is the bottleneck, not the code.** Small API/HTML calls are instant,
  but bulk file download from HF measured **~0.26 MB/s** unauthenticated. UltraFineWeb
  shards are 1.3 GB each (2048 of them) → 1–2 B tokens would take 10–20 h at that rate.
  `HF_TOKEN` lifts the throttle. I switched prep to **direct parquet-shard streaming**
  (`--ufw-shards N`) so it bypasses the very slow full-dataset resolution (which never
  returned in 90 s). Logic is solid; it's a pipe-size problem.

## Files
- **Changed:** `embeddings.py` (factorized `InputEmbedding`), `encoder.py` (passes
  `d_embed`), `mlm.py` (ALBERT `MLMHead` + wiring), `normalization.py` + `blocks.py`
  (earlier bug fixes).
- **New:** `data_prep.py` (resumable stream→memmap, direct-shard mode), `data.py`
  (memmap block dataset + train/val split), `train.py` (loop, AMP bf16, AdamW,
  warmup→cosine, grad-accum, eval, atomic checkpoint/resume; importable + CLI).
- **Wired:** `BERT.ipynb` (config → tokenizer+[MASK] → data → factorized model →
  train → masked-token inspection).
- `.gitignore` now excludes `data/`, `ckpt/`, `*.bin`.

## Notes / smaller caveats
- On resume, model+optimizer+scheduler+step+RNG are restored (loss continues
  smoothly). The data-shuffle position is *not* exactly restored (reshuffles) — fine
  for MLM. Keep `--max-steps` constant across resumes so the LR schedule stays continuous.
- OLMo has no `[MASK]`; `build_tokenizer` adds one (id 50280, vocab→50281). The memmap
  stores only real tokens (< 50280); `[MASK]` is injected at train time. uint16-safe.
- Packing is contiguous (docs joined by eos) → every block is full → no padding → no
  attention mask needed. This also sidesteps the current MHA having no key-padding mask.

## Log (oldest→newest)
- Step 1/2 done + verified (factorized embedding + ALBERT head; 22.7% embedding).
- Step 3 done + verified (resumable prep; byte-identical after simulated drop).
- Diagnosed slow UltraFineWeb: it's HF bulk bandwidth (~0.26 MB/s), not the pipeline.
  Added direct-shard streaming to sidestep full-dataset resolution.
- Step 4 done + verified (train loop: loss↓, masked_acc→1.0, resume continues).
- GPU confirmed available (3050 Ti); measured VRAM fit table above; full model trains bf16.
- Wired BERT.ipynb; launched a background resumable prep (crawling at ~0.26 MB/s).
