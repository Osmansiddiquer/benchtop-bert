"""Before/after comparison for the continuation fine-tune (greedy, identical prompts)."""
import torch
from mini_enc_transformer.training.pretrain import build_tokenizer, build_model


class C:
    d_model, d_k, d_v, n_heads, n_layers, d_embed = 768, 64, 64, 4, 4, 128


tk, ids = build_tokenizer("allenai/OLMo-1B-hf")
SEQ, NM = 128, 3
MASK, EOS = ids["mask_id"], tk.eos_token_id


def load(path):
    m = build_model(C(), ids)
    m.load_state_dict(torch.load(path, map_location="cpu")["model"])
    m.eval()
    return m


@torch.no_grad()
def gen(m, prompt, n=30):
    toks = tk(prompt, add_special_tokens=False)["input_ids"]
    stopped = None
    for _ in range(n):
        ctx = toks[-(SEQ - NM):]
        lg = m(torch.tensor([ctx + [MASK] * NM]))["logits"][0, len(ctx)]
        nxt = int(lg.argmax())
        if nxt == EOS:
            stopped = "eos"
            break
        toks.append(nxt)
        if tk.decode([nxt]).strip() in {".", "!", "?"}:
            stopped = "punct"
            break
    return tk.decode(toks), (stopped or "budget"), len(toks)


PROMPTS = ["Once upon a time there was a little girl who", "The movie was",
           "The capital of France is", "I watched this film last night and"]

base, tuned = load("checkpoints/ckpt3/last.pt"), load("checkpoints/ckpt_autoreg/ckpt_autoreg.pt")
for p in PROMPTS:
    b, bs, bn = gen(base, p)
    t, ts, tn = gen(tuned, p)
    print(f"> {p}")
    print(f"  BEFORE [{bs:6s} {bn:3d} tok]: {b}")
    print(f"  AFTER  [{ts:6s} {tn:3d} tok]: {t}\n")
