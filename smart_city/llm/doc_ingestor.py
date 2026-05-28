# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Document ingestion pipeline for RAG knowledge base.

Walks a configured directory, chunks text/markdown/PDF files, embeds
each chunk, and upserts them into the Milvus vector store.
Called once during application startup.
"""

import json
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

_SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def _read_text(path: Path) -> str:
    """Extract plain text from a file.

    Supports .txt, .md natively; .pdf via pypdf if available.

    Args:
        path: Path to the file.

    Returns:
        Extracted text, or empty string on error.
    """
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Cannot read %s: %s", path, exc)
            return ""
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader  # pylint: disable=import-outside-toplevel

            reader = PdfReader(str(path))
            return "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        except ImportError:
            logger.warning(
                "pypdf not installed; skipping PDF %s. "
                "Install with: pip install pypdf",
                path,
            )
        except (OSError, ValueError, KeyError, AssertionError) as exc:
            # pypdf raises a wide range of types for malformed PDFs
            # (PyPdfError subclasses ValueError; encrypted PDFs raise
            # AssertionError; corrupt streams raise OSError/KeyError).
            logger.warning(
                "Failed to parse PDF %s: %s", path, exc, exc_info=True
            )
        return ""
    return ""


def _chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[str]:
    """Split text into overlapping word-boundary chunks.

    Args:
        text: Full document text.
        chunk_size: Approximate max characters per chunk.
        chunk_overlap: Character overlap between consecutive chunks.

    Returns:
        List of text chunks; empty list if text is blank.

    Raises:
        ValueError: If chunk_overlap >= chunk_size.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than "
            f"chunk_size ({chunk_size})"
        )
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # Break at the last whitespace before the char limit
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        chunks.append(text[start:end].strip())
        start += step
    return [c for c in chunks if c]


def _glob_files(root: Path, patterns: List[str]) -> List[Path]:
    """Return files under root matching any of the glob patterns.

    Args:
        root: Directory to search recursively.
        patterns: List of glob patterns (e.g. ``["**/*.txt", "**/*.md"]``).

    Returns:
        Sorted, deduplicated list of matching paths.
    """
    found: set[Path] = set()
    for pattern in patterns:
        # root.glob handles "**/*.pdf" natively; avoids lstrip char-set bug
        for p in root.glob(pattern):
            if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES:
                found.add(p)
    return sorted(found)


async def ingest_documents(
    docs_ingest_path: str,
    docs_glob: str,
    embedder,
    vector_store,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> int:
    """Ingest all documents from a directory into the Milvus vector store.

    Skips files whose content cannot be read. Non-fatal: errors are logged
    and the function continues to the next file.

    Args:
        docs_ingest_path: Root directory containing documents.
        docs_glob: Comma-separated glob patterns (e.g.
            ``"**/*.txt,**/*.md,**/*.pdf"``).
        embedder: ``Embedder`` instance for generating vectors.
        vector_store: Connected ``VectorStoreClient`` instance.
        chunk_size: Max characters per chunk.
        chunk_overlap: Character overlap between chunks.

    Returns:
        Total number of chunks successfully inserted.
    """
    root = Path(docs_ingest_path)
    if not root.exists():
        logger.info(
            "docs_ingest_path '%s' does not exist; skipping ingestion.",
            docs_ingest_path,
        )
        return 0

    patterns = [p.strip() for p in docs_glob.split(",") if p.strip()]
    files = _glob_files(root, patterns)
    if not files:
        logger.info(
            "No documents found in '%s' matching %s.",
            docs_ingest_path,
            patterns,
        )
        return 0

    logger.info(
        "Starting document ingestion: %d file(s) from '%s'.",
        len(files),
        docs_ingest_path,
    )

    total_inserted = 0
    import asyncio  # pylint: disable=import-outside-toplevel

    for file_path in files:
        try:
            text = await asyncio.to_thread(_read_text, file_path)
            if not text.strip():
                logger.debug("Skipping empty file: %s", file_path)
                continue

            chunks = _chunk_text(text, chunk_size, chunk_overlap)
            if not chunks:
                continue

            embeddings = await asyncio.to_thread(
                embedder.embed_batch, chunks
            )
            metadata = [
                json.dumps(
                    {"source": str(file_path), "chunk_index": i}
                )
                for i in range(len(chunks))
            ]
            await asyncio.to_thread(
                vector_store.insert_events, chunks, embeddings, metadata
            )
            total_inserted += len(chunks)
            logger.info(
                "Ingested %d chunks from %s.", len(chunks), file_path.name
            )
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            logger.error(
                "Failed to ingest %s: %s", file_path, exc, exc_info=True
            )

    logger.info(
        "Document ingestion complete: %d total chunks inserted.",
        total_inserted,
    )
    return total_inserted
