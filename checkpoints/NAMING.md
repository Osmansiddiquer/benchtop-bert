# What each checkpoint is

"v2" was overloaded three ways. Disambiguation:

| path | lineage | objective | status |
|---|---|---|---|
| `ckpt/`            | phase 1, WikiText-103            | MLM, scattered | done |
| `ckpt2/`           | phase 2, UltraFineWeb 2 epochs   | MLM, scattered | done |
| `ckpt3/`           | phase 3, UFW+TinyStories+IMDB    | MLM, scattered | **best baseline: SST-2 85.09%** |
| `ckpt_span/`       | span-masking test from ckpt2     | MLM, uniform spans 2-4 | abandoned; weights deleted, metrics kept |
| `ckpt_jepa/`       | latent-target (data2vec-style)   | JEPA, no token loss | **failed** — worse on both frozen-repr measures |
| `ckpt_jepa_imdb/`  | JEPA + IMDB anneal               | JEPA | benchmarked, done |
| `ckpt_autoreg/`    | generation branches from ckpt3   | MLM + continuation/induction | done |
| `ckpt_v2_init.pt`  | ckpt3 -> 6-layer transfer        | (weights only) | input to ckpt_v2 |
| `ckpt_v2/`         | **6 layers, d_ff 1792, GELU**    | **MLM, geometric spans** | **LIVE** |
| `ckpt_sst2_*/`     | SST-2 fine-tune heads            | classification | see results/ |

Unrelated to the above, `results/finetune_sst2_results_v1/v2/v3.json` number the
*fine-tuning runs*, not the architectures:
  v1 = ckpt2 backbone, inflated protocol (max over evals, selection on dev)
  v2 = ckpt2 backbone, clean protocol -> 81.88%
  v3 = ckpt3 backbone, clean protocol -> 85.67%
