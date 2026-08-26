"""Inspect the planned embedding and Qdrant indexing operation offline."""

from dataclasses import asdict

from main import DEFAULT_PDF, EMBEDDING_MODEL, build_chunks


VECTOR_SIZE = 1_024
DISTANCE_METRIC = "cosine"
COLLECTION_NAME = "tesla_chunks"


def main() -> None:
    chunks = build_chunks(DEFAULT_PDF)
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

    total_tokens = sum(chunk.token_count for chunk in chunks)
    page_count = len({chunk.page for chunk in chunks})
    payload_fields = tuple(asdict(chunks[0]).keys())

    print("Indexing preflight (no API calls)")
    print(f"Source: {DEFAULT_PDF.name}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Texts to embed: {len(chunks)} complete cleaned chunk texts")
    print(f"Total document tokens sent: {total_tokens:,}")
    print(f"Pages represented: {page_count}")
    print(f"Qdrant collection: {COLLECTION_NAME}")
    print(f"Vector configuration: {VECTOR_SIZE} dimensions, {DISTANCE_METRIC}")
    print(f"Payload fields: {', '.join(payload_fields)}")


if __name__ == "__main__":
    main()
