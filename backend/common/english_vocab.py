"""Common English vocabulary, used to tell everyday words ("ordinary", "forgot") from specific terms ("uptime", "SLA").

The word list is the whole-word part of the cross-encoder tokenizer's WordPiece vocabulary (about 20k words). The
reranker model is downloaded anyway, so no extra file ships with the app. When the file is not available the set is
empty and callers fall back to their strict behaviour (every unknown word counts as specific).
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

logger = logging.getLogger(__name__)

_WORD = re.compile(r"^[a-z]{2,}$")


@lru_cache(maxsize=1)
def common_words() -> frozenset[str]:
    try:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache

        from config.settings import get_settings

        model = get_settings().reranker_model
        path = try_to_load_from_cache(model, "vocab.txt")
        if not isinstance(path, str):
            path = hf_hub_download(model, "vocab.txt")
        with open(path, encoding="utf-8") as f:
            words = frozenset(w for w in (line.strip() for line in f) if _WORD.match(w))
        logger.info("English vocabulary loaded: %d words", len(words))
        return words
    except Exception as exc:   # offline / non-BERT reranker: strict behaviour
        logger.warning("English vocabulary unavailable (%s); undocumented-term guard stays strict", exc)
        return frozenset()


def is_common_word(word: str) -> bool:
    """True for an everyday English word, including simple inflections (faked -> fake, forgot, ordinary)."""
    vocab = common_words()
    w = word.lower()
    if not vocab:
        return False
    candidates = {w}
    for suffix, repl in (("s", ""), ("es", ""), ("ed", ""), ("ed", "e"), ("d", ""), ("ing", ""), ("ing", "e"),
                         ("ly", ""), ("er", ""), ("est", "")):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            candidates.add(w[: len(w) - len(suffix)] + repl)
    return any(c in vocab for c in candidates)
