"""mini-enc-transformer: a small encoder-only transformer trained on a laptop GPU.

Convenience re-exports so notebooks can do:

    from mini_enc_transformer import build_tokenizer, build_model, Encoder

Resolved lazily (PEP 562): importing them eagerly would pull `mini_enc_transformer.training.pretrain`
into sys.modules before `python -m mini_enc_transformer.training.pretrain` executes it, which makes
runpy warn about unpredictable behaviour.
"""
import importlib

_LAZY = {
    "Encoder": "mini_enc_transformer.model.encoder",
    "BertForMaskedLM": "mini_enc_transformer.model.mlm",
    "MLMHead": "mini_enc_transformer.model.mlm",
    "mask_tokens": "mini_enc_transformer.model.mlm",
    "IGNORE_INDEX": "mini_enc_transformer.model.mlm",
    "PackedMemmapDataset": "mini_enc_transformer.data.dataset",
    "MixtureDataset": "mini_enc_transformer.data.dataset",
    "build_tokenizer": "mini_enc_transformer.training.pretrain",
    "build_model": "mini_enc_transformer.training.pretrain",
    "evaluate": "mini_enc_transformer.training.pretrain",
}
__all__ = sorted(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'mini_enc_transformer' has no attribute {name!r}")


def __dir__():
    return __all__
