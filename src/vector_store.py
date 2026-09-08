"""Create, populate, verify, and search the local Qdrant collection."""

from dataclasses import asdict
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from chunk import Chunk
from config import (
    COLLECTION_NAME,
    QDRANT_PATH,
    UPLOAD_BATCH_SIZE,
    VECTOR_SIZE,
)


def point_id_for_chunk(chunk_id: str) -> str:
    """Create a stable Qdrant-compatible UUID from a readable chunk ID."""
    return str(uuid5(NAMESPACE_URL, chunk_id))


def create_or_validate_collection(client: QdrantClient) -> None:
    """Create the collection, or reject an incompatible existing collection."""
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


def upload_chunks(
    client: QdrantClient,
    chunks: list[Chunk],
    vectors: list[list[float]],
) -> None:
    """Upsert chunk vectors and payloads in bounded batches."""
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


def verify_index(client: QdrantClient, chunks: list[Chunk]) -> None:
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
    if (
        not isinstance(sample_point.vector, list)
        or len(sample_point.vector) != VECTOR_SIZE
    ):
        raise RuntimeError("Stored sample vector has the wrong dimensions")

    print(f"Verified {point_count} stored points.")
    print(f"Sample point matches source chunk: {sample_chunk.chunk_id}")


def search_vectors(
    query_vectors: list[list[float]],
    top_k: int,
) -> list[list[models.ScoredPoint]]:
    """Search several query vectors while opening local Qdrant only once."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if any(len(vector) != VECTOR_SIZE for vector in query_vectors):
        raise ValueError(f"Every query vector must have {VECTOR_SIZE} dimensions")

    client = QdrantClient(path=QDRANT_PATH)
    try:
        return [
            client.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                with_payload=True,
                limit=top_k,
            ).points
            for vector in query_vectors
        ]
    finally:
        client.close()


def search_points(
    query_vector: list[float],
    top_k: int,
) -> list[models.ScoredPoint]:
    """Search one query vector against the local collection."""
    return search_vectors([query_vector], top_k)[0]
