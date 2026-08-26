"""Run and inspect one evaluation query against the local Qdrant index."""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from embed import embed_query
from evaluate import DEFAULT_DATASET, EvalExample, load_eval_dataset, normalize_for_match
from index_preflight import COLLECTION_NAME
from index_qdrant import QDRANT_PATH
from main import EMBEDDING_MODEL, PROJECT_ROOT


TOP_K = 5
PREVIEW_CHARS = 280


def select_example(examples: list[EvalExample], example_id: str) -> EvalExample:
    for example in examples:
        if example.id == example_id:
            return example
    available_ids = ", ".join(example.id for example in examples)
    raise ValueError(f"Unknown example ID {example_id!r}. Available IDs: {available_ids}")


def print_query_scope(example: EvalExample) -> None:
    print("Manual retrieval query")
    print(f"Example ID: {example.id}")
    print(f"Question sent to Voyage: {example.question}")
    print("Items sent: 1 query")
    print(f"Embedding model/input type: {EMBEDDING_MODEL} / query")
    print(f"Expected pages: {list(example.expected_pages)}")
    print(f"Qdrant collection: {COLLECTION_NAME} at {QDRANT_PATH}")
    print(f"Results requested: top {TOP_K}")


def inspect_results(example: EvalExample, points: list[object]) -> None:
    expected_pages = set(example.expected_pages)

    for rank, point in enumerate(points, start=1):
        payload = point.payload or {}
        page = payload.get("page")
        chunk_text = str(payload.get("text", ""))
        normalized_text = normalize_for_match(chunk_text)
        matched_phrases = [
            phrase
            for phrase in example.expected_phrases
            if normalize_for_match(phrase) in normalized_text
        ]
        preview = " ".join(chunk_text.split())[:PREVIEW_CHARS]

        print(f"\nRank {rank} | score={point.score:.4f} | page={page}")
        print(f"Chunk ID: {payload.get('chunk_id', '<missing>')}")
        print(f"Expected-page hit: {page in expected_pages}")
        print(f"Expected phrases found: {matched_phrases or 'none'}")
        print(f"Preview: {preview}")

    retrieved_pages = [
        point.payload.get("page")
        for point in points
        if point.payload is not None
    ]
    hit = bool(expected_pages & set(retrieved_pages))
    print(f"\nPage Hit@{TOP_K}: {hit}")


def retrieve_example(example: EvalExample) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    query_vector = embed_query(example.question, EMBEDDING_MODEL)

    client = QdrantClient(path=QDRANT_PATH)
    try:
        points = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            with_payload=True,
            limit=TOP_K,
        ).points
    finally:
        client.close()

    inspect_results(example, points)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview or execute one evaluation retrieval query."
    )
    parser.add_argument(
        "--example-id",
        default="model3_paid_reservations",
        help="Stable ID from eval_dataset.json.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Send one query to Voyage and search the local Qdrant collection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    example = select_example(load_eval_dataset(DEFAULT_DATASET), args.example_id)
    print_query_scope(example)

    if not args.execute:
        print("Dry run only. Re-run with --execute to perform retrieval.")
        return

    retrieve_example(example)


if __name__ == "__main__":
    main()
