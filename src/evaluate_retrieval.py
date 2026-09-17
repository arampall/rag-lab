"""Evaluate all golden questions against the local Qdrant index."""

from __future__ import annotations

import argparse

import voyageai
from dotenv import load_dotenv

from config import (
    COLLECTION_NAME,
    DEFAULT_DATASET,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    PROJECT_ROOT,
    QDRANT_PATH,
    VECTOR_SIZE,
)
from embed import embed_queries
from evaluate import (
    EvalExample,
    load_eval_dataset,
    normalize_for_match,
    score_page_hit_rate,
)
from vector_store import search_vectors


BASELINE_TOP_K = DEFAULT_TOP_K
EXPERIMENT_TOP_K = BASELINE_TOP_K + 1
EVALUATION_CUTOFFS = (BASELINE_TOP_K, EXPERIMENT_TOP_K)
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
    print(f"Results retrieved per query: {EXPERIMENT_TOP_K}")
    print(f"Evaluation cutoffs: {list(EVALUATION_CUTOFFS)}")
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


def score_phrase_hits(
    examples: list[EvalExample],
    points_by_id: dict[str, list[object]],
    top_k: int,
) -> tuple[float, dict[str, bool]]:
    hits_by_id = {
        example.id: phrase_hit(
            example,
            points_by_id[example.id][:top_k],
        )
        for example in examples
    }

    hit_count = sum(hits_by_id.values())
    return hit_count / len(examples), hits_by_id


def print_example_results(
    example: EvalExample,
    points: list[object],
) -> None:
    expected_pages = set(example.expected_pages)

    print(f"\n{example.id}")
    print(f"Inspecting top {len(points)} results")

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


def run_evaluation(examples: list[EvalExample]) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    embedding = embed_queries(
        [example.question for example in examples],
        EMBEDDING_MODEL,
    )
    if any(len(vector) != VECTOR_SIZE for vector in embedding.vectors):
        raise RuntimeError(f"Voyage did not return {VECTOR_SIZE}-dimension vectors")

    ranked_points = search_vectors(embedding.vectors, EXPERIMENT_TOP_K)
    points_by_id = {
        example.id: points
        for example, points in zip(examples, ranked_points, strict=True)
    }

    ranked_pages_by_id = {
        example_id: [
            point.payload.get("page")
            for point in points
            if point.payload is not None
        ]
        for example_id, points in points_by_id.items()
    }
    print("\nRetrieval comparison")

    for top_k in EVALUATION_CUTOFFS:

        page_hit_rate, page_results = score_page_hit_rate(
            examples,
            ranked_pages_by_id,
            top_k=top_k,
        )

        page_hit_by_id = {result.example_id: result.hit for result in page_results}
        page_hits = sum(page_hit_by_id.values())

        phrase_hit_rate, phrase_hits_by_id = score_phrase_hits(
            examples,
            points_by_id,
            top_k,
        )

        phrase_hits = sum(phrase_hits_by_id.values())

        print(f"\nTop {top_k}")
        print(
            f"Page Hit@{top_k}: {page_hit_rate:.1%} "
            f"({page_hits}/{len(examples)})"
        )

        print(
            f"Phrase Hit@{top_k}: {phrase_hit_rate:.1%} "
            f"({phrase_hits}/{len(examples)})"
        )

    print(f"Voyage billed query tokens: {embedding.total_tokens}")

    _, baseline_page_results = score_page_hit_rate(
        examples,
        ranked_pages_by_id,
        top_k=BASELINE_TOP_K,
    )

    _, experiment_page_results = score_page_hit_rate(
        examples,
        ranked_pages_by_id,
        top_k=EXPERIMENT_TOP_K,
    )

    baseline_hits = {
        result.example_id: result.hit
        for result in baseline_page_results
    }

    baseline_misses = [
        example
        for example in examples
        if not baseline_hits[example.id]
    ]

    print("\nBaseline misses, including rank 6 evidence")

    for example in baseline_misses:
        print_example_results(
            example,
            points_by_id[example.id],
        )

    experiment_hits = {
        result.example_id: result.hit
        for result in experiment_page_results
    }

    new_hits = [
        example.id
        for example in examples
        if not baseline_hits[example.id]
        and experiment_hits[example.id]
    ]

    print(f"New page hits at rank 6: {new_hits or 'none'}")


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
