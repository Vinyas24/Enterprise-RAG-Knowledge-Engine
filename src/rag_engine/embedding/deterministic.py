"""Deterministic, dependency-free embeddings for development and tests.

This is deliberately not a semantic model. It is a feature-hashing bag of words
and character n-grams: identical text always produces the same vector, texts
sharing words score higher by cosine similarity, and nothing leaves the
process. That is enough to exercise the vector pipeline end to end without an
API key, and it keeps CI hermetic.

Production uses the OpenAI-compatible provider; the interface makes the swap a
configuration change.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections.abc import Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NGRAM_PREFIX = "w:"


class DeterministicEmbeddingProvider:
    """Hash text features into a fixed-size unit vector."""

    name = "deterministic"

    def __init__(self, *, dimensions: int) -> None:
        if dimensions < 8:
            raise ValueError("deterministic embeddings need at least 8 dimensions")
        self.dimensions = dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    async def close(self) -> None:
        return None

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = _TOKEN_RE.findall(text.lower())
        features = list(tokens)
        features.extend(f"{left} {right}" for left, right in itertools.pairwise(tokens))

        for feature in features:
            digest = hashlib.blake2b(f"{_NGRAM_PREFIX}{feature}".encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimensions
            # A second bit decides the sign. Signed hashing keeps collisions
            # from systematically inflating similarity.
            vector[index] += 1.0 if (value >> 63) & 1 else -1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            # Empty or punctuation-only text still needs a unit vector so the
            # vector store never receives an all-zero embedding.
            digest = hashlib.blake2b(text.encode(), digest_size=8).digest()
            vector[int.from_bytes(digest, "big") % self.dimensions] = 1.0
            return vector
        return [value / norm for value in vector]
