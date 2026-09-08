"""Preview or execute document embedding and local Qdrant indexing."""

import argparse
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from chunk import Chunk
from config import (
    COLLECTION_NAME,
    DEFAULT_PDF,
    DISTANCE_METRIC,
    EMBEDDING_MODEL,
    PROJECT_ROOT,
    QDRANT_PATH,
    VECTOR_SIZE,
)
from embed import embed_chunks
from pipeline import build_chunks
from vector_store import (
    create_or_validate_collection,
    upload_chunks,
    verify_index,
)


def validate_chunks(chunks: list[Chunk]) -> None:
    """Reject an invalid indexing payload before any external call."""
    if not chunks:
        raise RuntimeError("No chunks were produced")

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise RuntimeError("Chunk IDs must be unique before indexing")

    for chunk in chunks:
        if not chunk.text.strip():
            raise RuntimeError(f"Chunk {chunk.chunk_id} has no text")
        if chunk.page < 1 or chunk.token_count < 1:
            raise RuntimeError(f"Chunk {chunk.chunk_id} has invalid metadata")


def print_execution_scope(chunks: list[Chunk], qdrant_path: Path) -> None:
    """Describe the complete embedding and indexing scope."""
    page_count = len({chunk.page for chunk in chunks})
    payload_fields = tuple(asdict(chunks[0]).keys())

    print("Qdrant indexing plan")
    print(f"Source: {DEFAULT_PDF.name}")
    print(f"Texts sent to Voyage: {len(chunks)} complete cleaned chunk texts")
    print(f"Document tokens sent: {sum(chunk.token_count for chunk in chunks):,}")
    print(f"Pages represented: {page_count}")
    print(f"Embedding model/input type: {EMBEDDING_MODEL} / document")
    print(f"Qdrant location: {qdrant_path}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Vector configuration: {VECTOR_SIZE} dimensions, {DISTANCE_METRIC}")
    print(f"Payload fields: {', '.join(payload_fields)}")


def execute_indexing(chunks: list[Chunk]) -> None:
    """Embed chunks and create or update the local collection."""
    client = QdrantClient(path=QDRANT_PATH)
    try:
        create_or_validate_collection(client)
        embedding = embed_chunks(chunks, EMBEDDING_MODEL)
        if any(len(vector) != VECTOR_SIZE for vector in embedding.vectors):
            raise RuntimeError(f"Voyage did not return {VECTOR_SIZE}-dimension vectors")
        upload_chunks(client, chunks, embedding.vectors)
        verify_index(client, chunks)
        print(f"Voyage billed tokens: {embedding.total_tokens:,}")
    finally:
        client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview or execute Voyage embedding and local Qdrant indexing."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Make the paid Voyage request and create/update the local collection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = build_chunks(DEFAULT_PDF)
    validate_chunks(chunks)
    print_execution_scope(chunks, QDRANT_PATH)

    if not args.execute:
        print("Dry run only. Re-run with --execute to perform indexing.")
        return

    load_dotenv(PROJECT_ROOT / ".env")
    execute_indexing(chunks)


if __name__ == "__main__":
    main()
