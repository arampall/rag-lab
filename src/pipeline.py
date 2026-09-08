"""Compose PDF extraction, cleaning, and chunking."""

from pathlib import Path

import voyageai

from chunk import Chunk, chunk_page
from config import EMBEDDING_MODEL
from extract import clean_text, extract_pages


def build_chunks(pdf_path: Path) -> list[Chunk]:
    """Extract, clean, and chunk every page in a PDF."""
    tokenizer = voyageai.Client().tokenizer(EMBEDDING_MODEL)
    chunks: list[Chunk] = []

    for page_number, raw_page_text in extract_pages(pdf_path):
        chunks.extend(
            chunk_page(
                page_text=clean_text(raw_page_text),
                page_number=page_number,
                source=pdf_path.name,
                tokenizer=tokenizer,
            )
        )

    return chunks
