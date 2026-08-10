"""BERT-style masked language modeling: token masking + MLM head + model wrapper.

This is the self-supervised pretraining objective from Devlin et al. (2018).
Because the encoder is bidirectional (no causal mask), we cannot train it with
next-token prediction -- every token would trivially see its own answer. Instead
we corrupt a fraction of the input and ask the model to reconstruct the originals
from both-sided context.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from mini_enc_transformer.model_v2.encoder import Encoder

# torch.nn.functional.cross_entropy ignores targets equal to this by default.
IGNORE_INDEX = -100


def _select_spans(input_ids, protected, mlm_probability, span_min, span_max, generator,
                  dist="uniform", geom_p=0.2):
    """Select ~`mlm_probability` of positions as contiguous spans.

    SpanBERT-style (Joshi et al. 2020). Scattered single-token masking is often
    solvable from immediate neighbours -- local morphology and collocations give the
    answer away. Masking a whole span removes that crutch and forces longer-range
    context. The *budget* is identical to random masking (same fraction of tokens
    scored), so the two differ only in how the masked positions are arranged.

    dist="geometric" reproduces SpanBERT's actual sampler: l ~ Geo(p), clipped to
    [span_min, span_max]. With p=0.2 and a clip of 10 the mean is ~3.8. The point is
    the long tail -- uniform 2..4 never produces a genuinely hard case, whereas the
    geometric occasionally masks 8-10 contiguous tokens, which is what forces
    long-range inference rather than local smoothing.
    """
    B, T = input_ids.shape
    selected = torch.zeros_like(input_ids, dtype=torch.bool)
    budget = int(round(mlm_probability * T))
    if budget == 0:
        return selected
    if dist == "geometric":
        # inverse-CDF sample of a geometric on {1,2,...}, then clip into range
        u = torch.rand((B, budget), generator=generator, device=input_ids.device)
        g = torch.floor(torch.log1p(-u) / math.log1p(-geom_p)) + 1
        lengths = g.clamp(span_min, span_max).long()
    else:
        lengths = torch.randint(span_min, span_max + 1, (B, budget), generator=generator,
                                device=input_ids.device)
    starts = torch.randint(0, T, (B, budget), generator=generator, device=input_ids.device)
    for b in range(B):
        placed, attempt = 0, 0
        # Cap attempts: once the sequence is crowded, random starts mostly collide.
        while placed < budget and attempt < budget * 4:
            i = attempt % budget
            s = int(starts[b, i])
            L = int(lengths[b, i])
            e = min(s + L, T)
            window = selected[b, s:e] | protected[b, s:e]
            if window.any():                      # overlaps an existing span or a special token
                attempt += 1
                starts[b, i] = torch.randint(0, T, (1,), generator=generator,
                                             device=input_ids.device)[0]
                continue
            take = min(e - s, budget - placed)    # never exceed the token budget
            selected[b, s:s + take] = True
            placed += take
            attempt += 1
    return selected


def mask_tokens(
    input_ids: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    special_token_ids=(),
    pad_token_id: int = None,
    mlm_probability: float = 0.15,
    mask_prob: float = 0.8,
    random_prob: float = 0.1,
    generator: torch.Generator = None,
    span_min: int = 1,
    span_max: int = 1,
    span_dist: str = "uniform",
    geom_p: float = 0.2,
):
    """Apply BERT's 80/10/10 masking to a batch of token ids.

    Of the ~`mlm_probability` (15%) of positions selected for prediction:
      - `mask_prob` (80%) are replaced with [MASK],
      - `random_prob` (10%) are replaced with a random token,
      - the rest (10%) are left unchanged.
    The 10% random / 10% unchanged split exists so the model cannot simply learn
    "output whatever [MASK] maps to" -- it must build a real representation of
    every position, since any position might be the one being scored.

    Returns (masked_input_ids, labels). `labels` holds the original id at scored
    positions and IGNORE_INDEX everywhere else, so the loss only sees the ~15%.
    Neither returned tensor aliases `input_ids`.
    """
    if not 0.0 <= mask_prob + random_prob <= 1.0:
        raise ValueError("mask_prob + random_prob must lie in [0, 1]")

    device = input_ids.device
    labels = input_ids.clone()
    masked_input = input_ids.clone()

    # Never corrupt or score special tokens ([CLS], [SEP], ...) or padding.
    protected = torch.zeros_like(input_ids, dtype=torch.bool)
    for tid in special_token_ids:
        protected |= input_ids == tid
    if pad_token_id is not None:
        protected |= input_ids == pad_token_id

    if span_max > 1:
        selected = _select_spans(input_ids, protected, mlm_probability,
                                 span_min, span_max, generator, span_dist, geom_p)
    else:
        prob_matrix = torch.full(input_ids.shape, mlm_probability, device=device)
        prob_matrix.masked_fill_(protected, 0.0)
        selected = torch.bernoulli(prob_matrix, generator=generator).bool()

    # Positions we don't score contribute nothing to the loss.
    labels[~selected] = IGNORE_INDEX

    # 80%: replace with [MASK].
    mask_replace = (
        torch.bernoulli(torch.full(input_ids.shape, mask_prob, device=device), generator=generator).bool()
        & selected
    )
    masked_input[mask_replace] = mask_token_id

    # 10%: replace with a random token. `random_prob` is unconditional over the
    # selected set, so among the (1 - mask_prob) not sent to [MASK] we draw at the
    # conditional rate random_prob / (1 - mask_prob).
    remaining = selected & ~mask_replace
    if mask_prob < 1.0:
        cond_random = random_prob / (1.0 - mask_prob)
        random_replace = (
            torch.bernoulli(torch.full(input_ids.shape, cond_random, device=device), generator=generator).bool()
            & remaining
        )
        random_tokens = torch.randint(
            vocab_size, input_ids.shape, dtype=input_ids.dtype, device=device, generator=generator
        )
        masked_input[random_replace] = random_tokens[random_replace]

    # The rest of `remaining` keeps its original token (already in masked_input).
    return masked_input, labels


class MLMHead(nn.Module):
    """Maps encoder hidden states (d_model) to per-token vocabulary logits.

    ALBERT's factorized head, mirroring the factorized input embedding: a
    transform (dense: d_model -> d_embed, GELU, LayerNorm over d_embed) then a
    decoder projecting d_embed -> vocab. The decoder weight is tied to the
    V x d_embed lookup table (see `tie_to`), leaving only the per-token bias to
    learn. The input projection (d_embed -> d_model) and this head's `dense`
    (d_model -> d_embed) are separate learned matrices, not transposes.
    """

    def __init__(self, d_model: int, d_embed: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(d_model, d_embed)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_embed)
        self.decoder = nn.Linear(d_embed, vocab_size, bias=True)

    def tie_to(self, embedding_weight: nn.Parameter):
        # embedding_weight: (vocab_size, d_embed) -- same shape as decoder.weight,
        # so the input and output token representations share one matrix.
        if embedding_weight.shape != self.decoder.weight.shape:
            raise ValueError(
                f"cannot tie: embedding {tuple(embedding_weight.shape)} "
                f"!= decoder {tuple(self.decoder.weight.shape)}"
            )
        self.decoder.weight = embedding_weight

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.norm(self.act(self.dense(hidden))))


class BertForMaskedLM(nn.Module):
    """Bidirectional encoder + MLM head, trained to reconstruct masked tokens."""

    def __init__(self, vocab_size: int, d_model: int, d_k: int, d_v: int, n_heads: int,
                 n_layers: int, d_ff=None, pad_id=None, d_embed: int = 128,
                 tie_weights: bool = True):
        super().__init__()
        # mask=False: BERT attends in both directions.
        self.encoder = Encoder(vocab_size, d_model, d_k, d_v, n_heads, n_layers,
                               d_ff=d_ff, mask=False, pad_id=pad_id, d_embed=d_embed)
        self.mlm_head = MLMHead(d_model, d_embed, vocab_size)
        if tie_weights:
            self.mlm_head.tie_to(self.encoder.embedding.weight)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        hidden = self.encoder(input_ids)          # (B, T, d_model)
        logits = self.mlm_head(hidden)            # (B, T, vocab_size)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )
        return {"loss": loss, "logits": logits}
