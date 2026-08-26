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
    if not query.strip():
        raise ValueError("Query must not be empty")

    client = voyageai.Client()
    result = client.embed([query], model=model, input_type="query")

    if len(result.embeddings) != 1:
        raise RuntimeError(
            f"Expected one query vector, received {len(result.embeddings)}"
        )

    return result.embeddings[0]

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
