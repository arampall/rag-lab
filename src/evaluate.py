"""Load and validate the retrieval evaluation dataset against its source PDF."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from extract import clean_text, extract_pages


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "src" / "eval_dataset.json"
DEFAULT_PDF = PROJECT_ROOT / "docs" / "Tesla_Inc.pdf"
REQUIRED_FIELDS = {
    "id",
    "question",
    "expected_answer",
    "expected_pages",
    "expected_phrases",
    "category",
    "notes",
}
ID_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


@dataclass(frozen=True)
class EvalExample:
    id: str
    question: str
    expected_answer: str
    expected_pages: tuple[int, ...]
    expected_phrases: tuple[str, ...]
    category: str
    notes: str


@dataclass(frozen=True)
class RetrievalResult:
    example_id: str
    expected_pages: tuple[int, ...]
    retrieved_pages: tuple[int, ...]
    hit: bool


def normalize_for_match(text: str) -> str:
    """Make source matching insensitive to case and whitespace artifacts."""
    return " ".join(text.casefold().split())


def _require_nonempty_string(record: dict[str, Any], field: str, index: int) -> str:
    value = record[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Record {index}: {field!r} must be a non-empty string")
    return value


def load_eval_dataset(dataset_path: Path) -> list[EvalExample]:
    """Load the JSON dataset and reject malformed or duplicate records."""
    with dataset_path.open(encoding="utf-8") as dataset_file:
        raw = json.load(dataset_file)

    if not isinstance(raw, list) or not raw:
        raise ValueError("Evaluation dataset must be a non-empty JSON list")

    examples: list[EvalExample] = []
    seen_ids: set[str] = set()

    for index, record in enumerate(raw, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index}: expected an object")

        missing = REQUIRED_FIELDS - record.keys()
        extra = record.keys() - REQUIRED_FIELDS
        if missing or extra:
            raise ValueError(
                f"Record {index}: missing fields {sorted(missing)}; "
                f"unexpected fields {sorted(extra)}"
            )

        example_id = _require_nonempty_string(record, "id", index)
        if not ID_PATTERN.fullmatch(example_id):
            raise ValueError(f"Record {index}: invalid snake_case id {example_id!r}")
        if example_id in seen_ids:
            raise ValueError(f"Record {index}: duplicate id {example_id!r}")
        seen_ids.add(example_id)

        pages = record["expected_pages"]
        if (
            not isinstance(pages, list)
            or not pages
            or any(isinstance(page, bool) or not isinstance(page, int) or page < 1 for page in pages)
        ):
            raise ValueError(
                f"Record {index}: expected_pages must contain positive integers"
            )
        if len(pages) != len(set(pages)):
            raise ValueError(f"Record {index}: expected_pages contains duplicates")

        phrases = record["expected_phrases"]
        if (
            not isinstance(phrases, list)
            or not phrases
            or any(not isinstance(phrase, str) or not phrase.strip() for phrase in phrases)
        ):
            raise ValueError(
                f"Record {index}: expected_phrases must contain non-empty strings"
            )

        examples.append(
            EvalExample(
                id=example_id,
                question=_require_nonempty_string(record, "question", index),
                expected_answer=_require_nonempty_string(
                    record, "expected_answer", index
                ),
                expected_pages=tuple(pages),
                expected_phrases=tuple(phrases),
                category=_require_nonempty_string(record, "category", index),
                notes=_require_nonempty_string(record, "notes", index),
            )
        )

    return examples


def score_page_hit_rate(
    examples: list[EvalExample],
    ranked_pages_by_id: dict[str, list[int]],
    top_k: int = 5,
) -> tuple[float, list[RetrievalResult]]:
    """Measure whether an expected page appears in each question's top-k results."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    expected_ids = {example.id for example in examples}
    result_ids = set(ranked_pages_by_id)
    if result_ids != expected_ids:
        missing = sorted(expected_ids - result_ids)
        unexpected = sorted(result_ids - expected_ids)
        raise ValueError(
            f"Retrieval result IDs do not match dataset IDs; "
            f"missing={missing}, unexpected={unexpected}"
        )

    results: list[RetrievalResult] = []
    for example in examples:
        retrieved_pages = tuple(ranked_pages_by_id[example.id][:top_k])
        hit = bool(set(example.expected_pages) & set(retrieved_pages))
        results.append(
            RetrievalResult(
                example_id=example.id,
                expected_pages=example.expected_pages,
                retrieved_pages=retrieved_pages,
                hit=hit,
            )
        )

    hit_rate = sum(result.hit for result in results) / len(results)
    return hit_rate, results


def validate_source_grounding(examples: list[EvalExample], pdf_path: Path) -> None:
    """Confirm declared pages exist and every expected phrase occurs on them."""
    normalized_pages = {
        page_number: normalize_for_match(clean_text(text))
        for page_number, text in extract_pages(pdf_path)
    }

    errors: list[str] = []
    for example in examples:
        invalid_pages = [
            page for page in example.expected_pages if page not in normalized_pages
        ]
        if invalid_pages:
            errors.append(f"{example.id}: pages do not exist: {invalid_pages}")
            continue

        declared_source = " ".join(
            normalized_pages[page] for page in example.expected_pages
        )
        for phrase in example.expected_phrases:
            if normalize_for_match(phrase) not in declared_source:
                errors.append(
                    f"{example.id}: phrase not found on declared pages: {phrase!r}"
                )

    if errors:
        raise ValueError("Source validation failed:\n- " + "\n- ".join(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate retrieval evaluation records against the source PDF."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_eval_dataset(args.dataset)
    validate_source_grounding(examples, args.pdf)
    phrase_count = sum(len(example.expected_phrases) for example in examples)
    print(
        f"Validated {len(examples)} examples and {phrase_count} source phrases "
        f"against {args.pdf.name}."
    )


if __name__ == "__main__":
    main()
