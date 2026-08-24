"""Connect PDF extraction, cleaning, chunking, and embedding inspection."""

import time
from pathlib import Path

import voyageai
from dotenv import load_dotenv

from chunk import Chunk, chunk_page
from extract import clean_text, extract_pages
from embed import Embedding, embed_chunks


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF = PROJECT_ROOT / "docs" / "Tesla_Inc.pdf"
EMBEDDING_MODEL = "voyage-4"


def build_chunks(pdf_path: Path) -> list[Chunk]:
    """Extract, clean, and chunk every page in a PDF."""
    client = voyageai.Client()
    tokenizer = client.tokenizer(EMBEDDING_MODEL)

    all_chunks: list[Chunk] = []

    for page_number, raw_page_text in extract_pages(pdf_path):
        cleaned_page_text = clean_text(raw_page_text)

        page_chunks: list[Chunk] = chunk_page(
            page_text=cleaned_page_text,
            page_number=page_number,
            source=pdf_path.name,
            tokenizer=tokenizer,
        )

        # Extend keeps all_chunks flat: list[Chunk], not list[list[Chunk]].
        all_chunks.extend(page_chunks)

    return all_chunks

def get_embeddings(chunks: list[Chunk], model: str) -> Embedding:
    start_time = time.perf_counter()
    result: Embedding =  embed_chunks(chunks, model)
    elasped_time = time.perf_counter() - start_time

    print(f"Chunks sent: {len(chunks)}")
    print(f"latency: {elasped_time}")
    print(f"Vector count: {len(result.vectors)}, dimensions: {len(result.vectors[0])}")
    print(f"Tokens Billed: {result.total_tokens}")
    return result

def main() -> None:
    # `.env` is convenient for local development. Production deployments should
    # inject environment variables through their platform or secret manager.
    load_dotenv(PROJECT_ROOT / ".env")

    chunks: list[Chunk] = build_chunks(DEFAULT_PDF)
    get_embeddings(chunks[:3])


if __name__ == "__main__":
    main()
