"""Evaluate grounded-generation behavior separately from retrieval."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

from dotenv import load_dotenv

from config import DEFAULT_DATASET, PROJECT_ROOT
from evaluate import EvalExample, load_eval_dataset, normalize_for_match
from generate import (
    FALLBACK_MESSAGE,
    GenerationResult,
    generate_result,
    require_api_keys,
)

CITATION_GROUP_PATTERN = re.compile(
    r"\[p\. \d+(?:,\s*p\. \d+)*\]"
)
CITATION_PAGE_PATTERN = re.compile(r"p\. (\d+)")


@dataclass(frozen=True)
class GenerationEvaluation:
    example_id: str
    context_has_answer: bool
    abstained: bool
    cited_pages: tuple[int, ...]
    citations_valid: bool
    behavior_passed: bool


def context_has_expected_phrase(
    example: EvalExample,
    result: GenerationResult,
) -> bool:
    normalized_chunks = [
        normalize_for_match(str((point.payload or {}).get("text", "")))
        for point in result.points
    ]

    return any(
        normalize_for_match(phrase) in chunk
        for phrase in example.expected_phrases
        for chunk in normalized_chunks
    )


def extract_cited_pages(answer: str) -> tuple[int, ...]:
    citation_groups = CITATION_GROUP_PATTERN.findall(answer)

    return tuple(
        int(page)
        for group in citation_groups
        for page in CITATION_PAGE_PATTERN.findall(group)
    )


def retrieved_pages(result: GenerationResult) -> set[int]:
    return {
        page
        for point in result.points
        if isinstance(
            page := (point.payload or {}).get("page"),
            int,
        )
    }


def evaluate_generation(
    example: EvalExample,
    result: GenerationResult,
) -> GenerationEvaluation:

    context_has_answer = context_has_expected_phrase(example, result)

    is_abstained: bool = result.answer == FALLBACK_MESSAGE

    cited_pages = extract_cited_pages(result.answer)

    citations_valid = (
        bool(cited_pages)
        and set(cited_pages).issubset(retrieved_pages(result))
    )

    if context_has_answer:
        behavior_passed = not is_abstained and citations_valid
    else:
        behavior_passed = is_abstained

    return GenerationEvaluation(
        example_id=example.id,
        context_has_answer=context_has_answer,
        abstained=is_abstained,
        cited_pages=cited_pages,
        citations_valid=citations_valid,
        behavior_passed=behavior_passed,
    )


def print_scope(examples: list[EvalExample]) -> None:
    print("Generation evaluation scope")
    print(f"Questions: {len(examples)}")
    print(f"Voyage requests: {len(examples)} query embeddings")
    print(f"Claude requests: {len(examples)}")
    print("Each Claude request receives one question and five retrieved chunks.")

    for example in examples:
        print(f"- {example.id}: {example.question}")


def print_example_evaluation(
    example: EvalExample,
    result: GenerationResult,
    evaluation: GenerationEvaluation,
) -> None:
    pages = [
        (point.payload or {}).get("page")
        for point in result.points
    ]

    print(f"\n{example.id}")
    print(f"Retrieved pages: {pages}")
    print(f"Context has expected phrase: {evaluation.context_has_answer}")
    print(f"Abstained: {evaluation.abstained}")
    print(f"Cited pages: {list(evaluation.cited_pages)}")
    print(f"Citations valid: {evaluation.citations_valid}")
    print(f"Grounding behavior passed: {evaluation.behavior_passed}")
    print(f"Expected answer: {example.expected_answer}")
    print(f"Generated answer: {result.answer}")
    print(
        f"Claude usage: {result.input_tokens} input, "
        f"{result.output_tokens} output tokens"
    )


def run_evaluation(examples: list[EvalExample]) -> None:
    evaluations: list[GenerationEvaluation] = []
    total_input_tokens = 0
    total_output_tokens = 0

    for index, example in enumerate(examples, start=1):
        print(
            f"\nRunning {index}/{len(examples)}: "
            f"{example.id}"
        )

        result = generate_result(example)
        evaluation = evaluate_generation(example, result)

        evaluations.append(evaluation)
        total_input_tokens += result.input_tokens
        total_output_tokens += result.output_tokens

        print_example_evaluation(example, result, evaluation)

    behavior_passes = sum(
        evaluation.behavior_passed
        for evaluation in evaluations
    )

    print("\nGeneration summary")
    print(
        f"Grounding behavior pass rate: "
        f"{behavior_passes / len(evaluations):.1%} "
        f"({behavior_passes}/{len(evaluations)})"
    )
    print(f"Claude input tokens: {total_input_tokens}")
    print(f"Claude output tokens: {total_output_tokens}")
    print("Answer correctness requires manual review above.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview grounded-generation evaluation."
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run retrieval and generation for every evaluation example.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_eval_dataset(DEFAULT_DATASET)
    print_scope(examples)

    if not args.execute:
        print("Dry run only. Re-run with --execute after reviewing the scope.")
        return

    load_dotenv(PROJECT_ROOT / ".env")
    require_api_keys()
    run_evaluation(examples)


if __name__ == "__main__":
    main()