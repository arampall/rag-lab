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
    elapsed_time = time.perf_counter() - start_time

    print(f"Chunks sent: {len(chunks)}")
    print(f"latency: {elapsed_time}")
    print(f"Vector count: {len(result.vectors)}, dimensions: {len(result.vectors[0])}")
    print(f"Tokens Billed: {result.total_tokens}")
    return result

def normalize_for_match(text: str) -> str:
    return " ".join(text.casefold().split())

def find_chunk(chunks: list[Chunk], phrase: str) -> Chunk:
    normalized_phrase = normalize_for_match(phrase)
    for chunk in chunks:
        if normalized_phrase.casefold() in normalize_for_match(chunk.text):
            return chunk

    raise ValueError(f"Could not find any chunk with the given phrase: {phrase}")


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    chunks: list[Chunk] = build_chunks(DEFAULT_PDF)

    # Testing
    sample_chunks = [
        find_chunk(chunks, "American multinational automotive"),
        find_chunk(chunks, "over 325,000 paid reservations"),
        find_chunk(chunks, "service its vehicles first through remote diagnosis")
    ]

    result = get_embeddings(sample_chunks, EMBEDDING_MODEL)

    for index, chunk in enumerate(sample_chunks):
        print(f"Vector {index} -> {chunk.chunk_id}")

    vector_dimension : int  = len(result.vectors[0])

    if not all(
        len(vector) == vector_dimension
        for vector in result.vectors
    ):
        raise RuntimeError("Embedding vectors have inconsistent dimensions.")


if __name__ == "__main__":
    main()
