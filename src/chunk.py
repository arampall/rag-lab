"""Split cleaned PDF pages into overlapping token-based chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    page: int
    token_count: int
    section: str | None = None
    subsection: str | None = None


def chunk_page(
    page_text: str,
    page_number: int,
    source: str,
    tokenizer: Tokenizer,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """Split one page into overlapping token windows."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size"
        )

    token_ids = tokenizer.encode(page_text).ids
    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0

    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunk_token_ids = token_ids[start:end]
        chunk_text = tokenizer.decode(chunk_token_ids)

        chunk_id = (
            f"{Path(source).stem}"
            f"-p{page_number:03d}"
            f"-c{chunk_index:03d}"
        )

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                source=source,
                page=page_number,
                token_count=len(chunk_token_ids),
            )
        )

        if end == len(token_ids):
            break

        start = end - chunk_overlap
        chunk_index += 1

    return chunks
