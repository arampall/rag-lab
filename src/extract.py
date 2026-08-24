"""Inspect page-by-page text extraction from a PDF using pypdf."""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

from pypdf import PdfReader


DEFAULT_PDF = Path("docs/Tesla_Inc.pdf")
DEFAULT_PAGES = (1, 3, 5, 8, 9, 40)


def extract_pages(pdf_path: Path) -> Iterator[tuple[int, str]]:
    """Yield each one-based page number and its extracted text."""
    reader = PdfReader(pdf_path)
    for page_number, page in enumerate(reader.pages, start=1):
        # Some PDFs contain image-only pages, for which extract_text returns None.
        yield page_number, page.extract_text() or ""


def print_pages(
    pdf_path: Path,
    requested_pages: Sequence[int],
    max_chars: int | None,
    clean: bool = False,
) -> None:
    """Print selected pages while preserving page boundaries."""
    pages = dict(extract_pages(pdf_path))
    print(f"PDF: {pdf_path}")
    print(f"Total pages: {len(pages)}")

    for page_number in requested_pages:
        if page_number not in pages:
            raise ValueError(
                f"Page {page_number} is outside the valid range 1-{len(pages)}."
            )

        text = pages[page_number]
        if clean:
            text = clean_text(text)

        displayed_text = text if max_chars is None else text[:max_chars]
        text_kind = "CLEANED" if clean else "RAW"
        print(
            f"\n{'=' * 20} PAGE {page_number} "
            f"({text_kind}, {len(text)} chars) {'=' * 20}"
        )
        print(displayed_text)
        if max_chars is not None and len(text) > max_chars:
            print(f"\n[truncated: showing {max_chars} of {len(text)} characters]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and print selected PDF pages with visible boundaries."
    )
    parser.add_argument("pdf", nargs="?", type=Path, default=DEFAULT_PDF)
    parser.add_argument(
        "--pages",
        nargs="+",
        type=int,
        default=DEFAULT_PAGES,
        help="One-based page numbers (default: 1 3 5 8 9 40).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=4_000,
        help="Characters printed per page; use 0 for the complete text.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Apply conservative whitespace cleaning before printing.",
    )
    return parser.parse_args()


def clean_text(page_text: str) -> str:
    """Normalize whitespace without guessing the document's structure."""
    text = page_text.replace("\u00a0", " ")

    cleaned_lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def main() -> None:
    args = parse_args()
    if not args.pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {args.pdf}")
    if args.max_chars < 0:
        raise ValueError("--max-chars must be zero or greater.")

    max_chars = None if args.max_chars == 0 else args.max_chars
    print_pages(args.pdf, args.pages, max_chars, clean=args.clean)


if __name__ == "__main__":
    main()
