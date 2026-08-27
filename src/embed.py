"""Generate Voyage document embeddings for chunks."""

from dataclasses import dataclass
from chunk import Chunk
import voyageai

@dataclass
class Embedding:
    vectors : list[list[float]]
    total_tokens: int

def embed_query(query: str, model: str) -> list[float]:
    """Embed one retrieval query and return its single vector."""
    return embed_queries([query], model).vectors[0]


def embed_queries(queries: list[str], model: str) -> Embedding:
    """Embed retrieval queries as one validated batch."""
    if not queries:
        raise ValueError("At least one query is required")
    if any(not query.strip() for query in queries):
        raise ValueError("Queries must not be empty")

    client = voyageai.Client()
    result = client.embed(queries, model=model, input_type="query")

    if len(result.embeddings) != len(queries):
        raise RuntimeError(
            f"Expected {len(queries)} query vectors, "
            f"received {len(result.embeddings)}"
        )

    return Embedding(
        vectors=result.embeddings,
        total_tokens=result.total_tokens,
    )

def embed_chunks(chunks: list[Chunk], model: str) -> Embedding:
    if not chunks:
        raise ValueError("Atleast one chunk required")
   
    client = voyageai.Client()

    result = client.embed(
        [chunk.text for chunk in chunks], 
        model=model, 
        input_type="document"
    )

    vectors = result.embeddings

    if len(vectors) != len(chunks):
        raise RuntimeError(f"Expected {len(chunks)} vectors, received {len(vectors)}")

    return Embedding(
        vectors=vectors,
        total_tokens=result.total_tokens
    )
