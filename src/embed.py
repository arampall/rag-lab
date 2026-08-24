"""Generate Voyage document embeddings for chunks."""

from dataclasses import dataclass
from chunk import Chunk
import voyageai

@dataclass
class Embedding:
    vectors : list[list[float]]
    total_tokens: int

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