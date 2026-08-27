"""Evaluate all golden questions against the local Qdrant index."""

from __future__ import annotations

import argparse

import voyageai
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from embed import embed_queries
from evaluate import (
    DEFAULT_DATASET,
    EvalExample,
    load_eval_dataset,
    normalize_for_match,
    score_page_hit_rate,
)
from index_preflight import COLLECTION_NAME, VECTOR_SIZE
from index_qdrant import QDRANT_PATH
from main import EMBEDDING_MODEL, PROJECT_ROOT


TOP_K = 5
PREVIEW_CHARS = 180


def count_query_tokens(examples: list[EvalExample]) -> int:
    tokenizer = voyageai.Client().tokenizer(EMBEDDING_MODEL)
    return sum(len(tokenizer.encode(example.question).ids) for example in examples)


def print_scope(examples: list[EvalExample]) -> None:
    print("Retrieval evaluation scope")
    print(f"Queries sent to Voyage: {len(examples)} complete question strings")
    print(f"Query tokens before Voyage instruction: {count_query_tokens(examples)}")
    print(f"Embedding model/input type: {EMBEDDING_MODEL} / query")
    print(f"Qdrant collection: {COLLECTION_NAME} at {QDRANT_PATH}")
    print(f"Results per query: {TOP_K}")
    for example in examples:
        print(f"- {example.id}: {example.question}")


def phrase_hit(example: EvalExample, points: list[object]) -> bool:
    normalized_chunks = [
        normalize_for_match(str((point.payload or {}).get("text", "")))
        for point in points
    ]
    return any(
        normalize_for_match(phrase) in chunk_text
        for phrase in example.expected_phrases
        for chunk_text in normalized_chunks
    )


def print_example_results(
    example: EvalExample,
    points: list[object],
    page_hit: bool,
) -> bool:
    expected_pages = set(example.expected_pages)
    has_phrase_hit = phrase_hit(example, points)
    print(f"\n{example.id}")
    print(f"Page Hit@{TOP_K}: {page_hit} | Phrase hit: {has_phrase_hit}")

    for rank, point in enumerate(points, start=1):
        payload = point.payload or {}
        page = payload.get("page")
        text = str(payload.get("text", ""))
        normalized_text = normalize_for_match(text)
        matches_phrase = any(
            normalize_for_match(phrase) in normalized_text
            for phrase in example.expected_phrases
        )
        preview = " ".join(text.split())[:PREVIEW_CHARS]
        print(
            f"  {rank}. score={point.score:.4f} page={page} "
            f"chunk={payload.get('chunk_id', '<missing>')} "
            f"expected_page={page in expected_pages} phrase={matches_phrase}"
        )
        print(f"     {preview}")

    return has_phrase_hit


def run_evaluation(examples: list[EvalExample]) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    embedding = embed_queries(
        [example.question for example in examples],
        EMBEDDING_MODEL,
    )
    if any(len(vector) != VECTOR_SIZE for vector in embedding.vectors):
        raise RuntimeError(f"Voyage did not return {VECTOR_SIZE}-dimension vectors")

    points_by_id: dict[str, list[object]] = {}
    client = QdrantClient(path=QDRANT_PATH)
    try:
        for example, vector in zip(examples, embedding.vectors, strict=True):
            points_by_id[example.id] = client.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                with_payload=True,
                limit=TOP_K,
            ).points
    finally:
        client.close()

    ranked_pages_by_id = {
        example_id: [
            point.payload.get("page")
            for point in points
            if point.payload is not None
        ]
        for example_id, points in points_by_id.items()
    }
    page_hit_rate, page_results = score_page_hit_rate(
        examples,
        ranked_pages_by_id,
        top_k=TOP_K,
    )
    page_hit_by_id = {result.example_id: result.hit for result in page_results}

    phrase_hits = sum(
        print_example_results(
            example,
            points_by_id[example.id],
            page_hit_by_id[example.id],
        )
        for example in examples
    )

    print("\nRetrieval summary")
    print(f"Page Hit@{TOP_K}: {page_hit_rate:.1%} ({sum(page_hit_by_id.values())}/{len(examples)})")
    print(f"Phrase hit rate: {phrase_hits / len(examples):.1%} ({phrase_hits}/{len(examples)})")
    print(f"Voyage billed query tokens: {embedding.total_tokens}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview or execute retrieval evaluation for all golden questions."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Embed all questions with Voyage and search the local collection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_eval_dataset(DEFAULT_DATASET)
    print_scope(examples)

    if not args.execute:
        print("Dry run only. Re-run with --execute to evaluate retrieval.")
        return

    run_evaluation(examples)


if __name__ == "__main__":
    main()
