"""Embed document chunks and store them in a local Qdrant collection."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv

from chunk import Chunk
from embed import embed_chunks
from index_preflight import COLLECTION_NAME, DISTANCE_METRIC, VECTOR_SIZE
from main import DEFAULT_PDF, EMBEDDING_MODEL, PROJECT_ROOT, build_chunks


QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"
UPLOAD_BATCH_SIZE = 64


def point_id_for_chunk(chunk_id: str) -> str:
    """Create a stable Qdrant-compatible UUID from a readable chunk ID."""
    return str(uuid5(NAMESPACE_URL, chunk_id))


def print_execution_scope(chunks: list[Chunk], qdrant_path: Path) -> None:
    print("Qdrant indexing plan")
    print(f"Texts sent to Voyage: {len(chunks)} complete cleaned chunk texts")
    print(f"Document tokens sent: {sum(chunk.token_count for chunk in chunks):,}")
    print(f"Embedding model/input type: {EMBEDDING_MODEL} / document")
    print(f"Qdrant location: {qdrant_path}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Vector configuration: {VECTOR_SIZE} dimensions, {DISTANCE_METRIC}")


def create_or_validate_collection(client: object) -> None:
    """Create the collection, or reject an incompatible existing collection."""
    from qdrant_client import models

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )
        return

    collection = client.get_collection(COLLECTION_NAME)
    vector_config = collection.config.params.vectors
    if isinstance(vector_config, dict):
        raise RuntimeError("Expected one unnamed vector configuration")
    if (
        vector_config.size != VECTOR_SIZE
        or vector_config.distance != models.Distance.COSINE
    ):
        raise RuntimeError(
            f"Existing collection {COLLECTION_NAME!r} has incompatible "
            f"vector configuration: {vector_config}"
        )


def upload_chunks(client: object, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    """Upsert chunk vectors and payloads in bounded batches."""
    from qdrant_client import models

    for start in range(0, len(chunks), UPLOAD_BATCH_SIZE):
        batch_chunks = chunks[start : start + UPLOAD_BATCH_SIZE]
        batch_vectors = vectors[start : start + UPLOAD_BATCH_SIZE]
        points = [
            models.PointStruct(
                id=point_id_for_chunk(chunk.chunk_id),
                vector=vector,
                payload=asdict(chunk),
            )
            for chunk, vector in zip(batch_chunks, batch_vectors, strict=True)
        ]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )


def verify_index(client: object, chunks: list[Chunk]) -> None:
    """Verify point count and one stored point against the source chunk."""
    point_count = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    ).count
    if point_count != len(chunks):
        raise RuntimeError(f"Expected {len(chunks)} points, found {point_count}")

    sample_chunk = chunks[0]
    stored_points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[point_id_for_chunk(sample_chunk.chunk_id)],
        with_payload=True,
        with_vectors=True,
    )
    if len(stored_points) != 1:
        raise RuntimeError("Could not retrieve the sample point")

    sample_point = stored_points[0]
    if sample_point.payload != asdict(sample_chunk):
        raise RuntimeError("Stored sample payload does not match its source chunk")
    if not isinstance(sample_point.vector, list) or len(sample_point.vector) != VECTOR_SIZE:
        raise RuntimeError("Stored sample vector has the wrong dimensions")

    print(f"Verified {point_count} stored points.")
    print(f"Sample point matches source chunk: {sample_chunk.chunk_id}")


def execute_indexing(chunks: list[Chunk]) -> None:
    """Perform the explicitly requested embedding and local indexing operation."""
    from qdrant_client import QdrantClient

    load_dotenv(PROJECT_ROOT / ".env")
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
    print_execution_scope(chunks, QDRANT_PATH)

    if not args.execute:
        print("Dry run only. Re-run with --execute to perform indexing.")
        return

    execute_indexing(chunks)


if __name__ == "__main__":
    main()
