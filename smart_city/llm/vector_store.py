# Created by Metrum AI for AMD

"""Milvus-backed vector store for crowd event RAG context."""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# pymilvus is an optional/lazy dependency — fall back to ``Exception`` so
# this module is still importable in environments without pymilvus, while
# still allowing callers to catch the narrowest meaningful type when it is
# available.
try:  # pragma: no cover - import shim
    from pymilvus.exceptions import MilvusException  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - import shim
    class MilvusException(Exception):  # type: ignore[no-redef]
        """Fallback used when pymilvus is unavailable."""

# Collection name used across all RAG operations
COLLECTION_NAME = "crowd_events"
EMBEDDING_DIM = 384


class VectorStoreClient:
    """Thin async wrapper around the Milvus collection for RAG.

    Provides insert and similarity-search helpers used by
    ConversationalQA and ReportGenerator.
    """

    def __init__(
        self,
        host: str = "milvus",
        port: int = 19530,
        collection_name: str = COLLECTION_NAME,
        embedding_dim: int = EMBEDDING_DIM,
    ) -> None:
        """Initialise the client (does not connect immediately).

        Args:
            host: Milvus server hostname.
            port: Milvus gRPC port.
            collection_name: Target Milvus collection.
            embedding_dim: Dimension of the embedding vectors.
        """
        self._host = host
        self._port = port
        self._collection_name = collection_name
        self._embedding_dim = embedding_dim
        self._collection: Optional[Any] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to Milvus and ensure collection exists.

        Raises:
            RuntimeError: If pymilvus is not installed or connection fails.
        """
        from pymilvus import connections

        connections.connect(host=self._host, port=self._port)
        self._ensure_collection()
        logger.info(
            "Connected to Milvus collection '%s'.",
            self._collection_name,
        )

    def drop_and_recreate(self) -> None:
        """Drop the collection and recreate it from scratch.

        Called at startup to ensure a clean slate and prevent duplicate
        chunks accumulating across container restarts.
        """
        try:
            from pymilvus import utility

            if utility.has_collection(self._collection_name):
                utility.drop_collection(self._collection_name)
                logger.info(
                    "Dropped Milvus collection '%s' for clean re-ingestion.",
                    self._collection_name,
                )
            self._collection = None
            self._ensure_collection()
        except (MilvusException, ImportError, RuntimeError, OSError) as exc:
            logger.warning(
                "drop_and_recreate failed: %s", exc, exc_info=True
            )

    def _ensure_collection(self) -> None:
        """Create the collection and index if not already present."""
        try:
            from pymilvus import (
                Collection,
                CollectionSchema,
                DataType,
                FieldSchema,
                utility,
            )

            if not utility.has_collection(self._collection_name):
                fields = [
                    FieldSchema(
                        name="id",
                        dtype=DataType.INT64,
                        is_primary=True,
                        auto_id=True,
                    ),
                    FieldSchema(
                        name="text",
                        dtype=DataType.VARCHAR,
                        max_length=4096,
                    ),
                    FieldSchema(
                        name="metadata",
                        dtype=DataType.VARCHAR,
                        max_length=2048,
                    ),
                    FieldSchema(
                        name="embedding",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=self._embedding_dim,
                    ),
                ]
                schema = CollectionSchema(
                    fields, description="Smart City crowd events"
                )
                col = Collection(name=self._collection_name, schema=schema)
                col.create_index(
                    field_name="embedding",
                    index_params={
                        "index_type": "HNSW",
                        "metric_type": "COSINE",
                        "params": {"M": 16, "efConstruction": 256},
                    },
                )
                logger.info(
                    "Created Milvus collection '%s'.",
                    self._collection_name,
                )

            self._collection = Collection(self._collection_name)
            if not self._collection.indexes:
                self._collection.create_index(
                    field_name="embedding",
                    index_params={
                        "index_type": "HNSW",
                        "metric_type": "COSINE",
                        "params": {"M": 16, "efConstruction": 256},
                    },
                )
            self._collection.load()
        except (MilvusException, ImportError, RuntimeError, OSError) as exc:
            logger.error(
                "_ensure_collection failed: %s", exc, exc_info=True
            )

    def disconnect(self) -> None:
        """Release the Milvus connection."""
        try:
            from pymilvus import connections

            connections.disconnect("default")
        except (MilvusException, ImportError, RuntimeError, OSError) as exc:
            logger.debug("Milvus disconnect failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def insert_events(
        self, texts: List[str], embeddings: List[List[float]],
        metadata: Optional[List[str]] = None,
    ) -> None:
        """Insert event texts and their embeddings into Milvus.

        Args:
            texts: Event description strings.
            embeddings: Corresponding float vectors.
            metadata: Optional JSON-serialised metadata per event.

        Raises:
            RuntimeError: If collection is not connected.
        """
        if self._collection is None:
            logger.warning("Milvus collection not connected; skipping insert.")
            return
        meta = metadata or ["{}"] * len(texts)
        self._collection.insert([texts, meta, embeddings])
        self._collection.flush()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve the top-k most similar events.

        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.
            expr: Optional Milvus boolean expression filter.

        Returns:
            List of dicts with keys 'text', 'metadata', 'score'.
        """
        if self._collection is None:
            return []

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
        results = self._collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["text", "metadata"],
        )

        hits = []
        for hit in results[0]:
            # pymilvus 2.4 Entity.get() accepts only the key; use or-default
            hits.append(
                {
                    "text": hit.entity.get("text") or "",
                    "metadata": hit.entity.get("metadata") or "{}",
                    "score": hit.score,
                }
            )
        return hits
