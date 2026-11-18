"""Pluggable embedding backends.

EMBEDDING_BACKEND=
  openai                -> OpenAI embeddings API (default; EMBEDDING_MODEL=text-embedding-3-small)
  sentence_transformers -> local/on-prem model (e.g. BAAI/bge-small-en-v1.5, all-MiniLM-L6-v2)
  local_hash            -> dependency-free hashed n-gram vectors (offline tests / air-gapped demos;
                           lexical, not truly semantic)

All backends return L2-normalised float vectors so cosine similarity == dot product.
The backend+model id is stored with every collection; mixing models in one collection
is refused (re-index instead).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np

from backend.common.text_utils import normalize_text, tokenize
from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class Embedder(ABC):
    model_id: str = "base"

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (matrix / norms).astype(np.float32)


class LocalHashEmbedder(Embedder):
    """Stateless hashed bag of stemmed words, word bigrams and character n-grams."""

    def __init__(self, dim: int = 2048) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self.model_id = f"local_hash-{dim}"
        self._words = HashingVectorizer(n_features=dim, analyzer=self._word_analyzer, alternate_sign=False,
                                        norm="l2")
        self._chars = HashingVectorizer(n_features=dim, analyzer="char_wb", ngram_range=(3, 5),
                                        alternate_sign=False, norm="l2", lowercase=True)

    @staticmethod
    def _word_analyzer(text: str) -> list[str]:
        toks = tokenize(text)
        return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        texts = [normalize_text(t) for t in texts]
        words = self._words.transform(texts).toarray()
        chars = self._chars.transform(texts).toarray()
        return self._normalize(np.hstack([words * 0.75, chars * 0.25]))


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str, batch_size: int = 64) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.model_id = f"sentence_transformers:{model_name}"
        name = model_name.lower()
        # Model-specific input formats (from the model cards): BGE English v1.5 uses a query instruction; E5 models
        # need "query: " / "passage: " prefixes; BGE-M3 and others take raw text.
        if "e5-" in name:
            self._query_prefix, self._doc_prefix = "query: ", "passage: "
        elif "bge-" in name and "-en" in name:
            self._query_prefix, self._doc_prefix = "Represent this sentence for searching relevant passages: ", ""
        else:
            self._query_prefix, self._doc_prefix = "", ""

    def _encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode([self._doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self._query_prefix + text])[0]


class OpenAIEmbedder(Embedder):
    def __init__(self, model_name: str, api_key: str, base_url: str = "", batch_size: int = 64) -> None:
        from openai import OpenAI

        if not api_key:
            raise RuntimeError("EMBEDDING_BACKEND=openai requires OPENAI_API_KEY in .env "
                               "(or set EMBEDDING_BACKEND=local_hash / sentence_transformers to run without it)")
        self.client = OpenAI(api_key=api_key, base_url=base_url or None)
        self.model_name = model_name
        self.batch_size = batch_size
        self.model_id = f"openai:{model_name}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t[:8000] or " " for t in texts[start:start + self.batch_size]]
            response = self.client.embeddings.create(model=self.model_name, input=batch)
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return self._normalize(np.asarray(vectors, dtype=np.float32))


def build_embedder(settings: Settings) -> Embedder:
    backend = settings.embedding_backend
    if backend == "openai":
        if settings.embedding_api_key:   # dedicated embedding credentials (default host: api.openai.com)
            api_key, base_url = settings.embedding_api_key, settings.embedding_base_url
        else:
            api_key, base_url = settings.openai_api_key, settings.embedding_base_url or settings.openai_base_url
        return OpenAIEmbedder(settings.embedding_model, api_key, base_url, settings.embedding_batch_size)
    if backend == "sentence_transformers":
        return SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_batch_size)
    if backend == "local_hash":
        return LocalHashEmbedder()
    raise ValueError(f"Unknown EMBEDDING_BACKEND '{backend}'")


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    embedder = build_embedder(get_settings())
    logger.info("Embedding backend: %s", embedder.model_id)
    return embedder
