# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Sentence-transformer embedding helper for RAG.

Supports two backends:
1. **HTTP** (preferred): calls a vLLM / TEI OpenAI-compatible
   ``POST /v1/embeddings`` endpoint.  Set ``base_url`` to e.g.
   ``http://vllm-embed:8000`` to enable this mode.  No GPU or model
   download is needed in the API container.
2. **Local** (fallback): loads the model via sentence-transformers
   directly in-process.  Used when ``base_url`` is empty.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder:
    """Dense text embedder backed by either a remote HTTP endpoint or a
    local sentence-transformers model.

    The HTTP backend delegates embedding computation to a running vLLM
    (or any OpenAI-compatible) embedding service, avoiding the need to
    download and serve the model inside the API container.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        base_url: Optional[str] = None,
    ) -> None:
        """Initialise the embedder.

        Args:
            model_name: HuggingFace model identifier.  For the HTTP
                backend this must match the ``--served-model-name``
                passed to the embedding service.
            base_url: Base URL of an OpenAI-compatible embedding service
                (e.g. ``http://vllm-embed:8000``).  When provided the
                HTTP backend is used and no local model is loaded.
        """
        self._model_name = model_name
        self._dim = 384
        self._model = None  # local sentence-transformers model

        # Normalise base_url: strip trailing slash and /v1 suffix so we
        # can always append /v1/embeddings uniformly.
        if base_url:
            _url = base_url.strip().rstrip("/")
            self._base_url: Optional[str] = (
                _url[:-3] if _url.endswith("/v1") else _url
            )
        else:
            self._base_url = None

        if self._base_url:
            logger.info(
                "Embedder: HTTP backend → %s  (model=%s)",
                self._base_url,
                self._model_name,
            )
        else:
            self._load_local()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_local(self) -> None:
        """Attempt to load the sentence-transformers model locally."""
        try:
            from sentence_transformers import (  # pylint: disable=import-outside-toplevel
                SentenceTransformer,
            )

            self._model = SentenceTransformer(self._model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
            logger.info(
                "Embedder: local model '%s' loaded (dim=%d).",
                self._model_name,
                self._dim,
            )
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; "
                "using zero-vector fallback."
            )
        except (OSError, RuntimeError, ValueError) as exc:
            logger.error("Embedder local load failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Return the embedding dimension."""
        return self._dim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Produce dense embeddings for a list of strings.

        Calls the HTTP backend if configured; otherwise uses the local
        sentence-transformers model.  Returns zero-vectors when both
        backends are unavailable.

        Args:
            texts: Input strings to embed.

        Returns:
            List of float vectors, one per input string.
        """
        if not texts:
            return []
        if self._base_url:
            return self._embed_http(texts)
        return self._embed_local(texts)

    # Alias used by doc_ingestor
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Alias for :meth:`embed`."""
        return self.embed(texts)

    def embed_one(self, text: str) -> List[float]:
        """Produce a single dense embedding.

        Args:
            text: Input string.

        Returns:
            Float vector of length ``dim``.
        """
        return self.embed([text])[0]

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _embed_http(self, texts: List[str]) -> List[List[float]]:
        """Call the remote embedding endpoint.

        Supports two response formats automatically:
        - **TEI** (``POST /embed``): ``[[float, …], …]``
        - **OpenAI-compatible** (``POST /v1/embeddings``):
          ``{"data": [{"index": i, "embedding": [float, …]}, …]}``

        Tries TEI format first (``/embed``), then falls back to the
        OpenAI format (``/v1/embeddings``) on failure.

        Args:
            texts: Input strings.

        Returns:
            List of float vectors.
        """
        import httpx  # pylint: disable=import-outside-toplevel

        def _try_tei() -> List[List[float]]:
            resp = httpx.post(
                f"{self._base_url}/embed",
                json={"inputs": texts},
                timeout=30.0,
            )
            resp.raise_for_status()
            result = resp.json()
            # TEI returns a list-of-lists directly
            if result and isinstance(result[0], list):
                return result
            raise ValueError("Unexpected TEI response shape")

        def _try_openai() -> List[List[float]]:
            resp = httpx.post(
                f"{self._base_url}/v1/embeddings",
                json={"model": self._model_name, "input": texts},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            data.sort(key=lambda d: d["index"])
            return [d["embedding"] for d in data]

        for fn, label in [(_try_tei, "TEI"), (_try_openai, "OpenAI")]:
            try:
                vecs = fn()
                if vecs and self._dim != len(vecs[0]):
                    self._dim = len(vecs[0])
                return vecs
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                logger.debug(
                    "HTTP embedding %s format failed: %s",
                    label, exc, exc_info=True,
                )

        raise RuntimeError(
            f"All HTTP embedding backends failed for {self._base_url}. "
            "Check that the embedding service is reachable."
        )

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        """Run the local sentence-transformers model.

        Args:
            texts: Input strings.

        Returns:
            List of float vectors; zero-vectors if model unavailable.
        """
        if self._model is None:
            return [[0.0] * self._dim for _ in texts]
        try:
            vecs = self._model.encode(texts, convert_to_numpy=True)
            return [v.tolist() for v in vecs]
        except (RuntimeError, ValueError, OSError) as exc:
            logger.error("Local embedding failed: %s", exc, exc_info=True)
            return [[0.0] * self._dim for _ in texts]
