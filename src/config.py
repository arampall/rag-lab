"""Shared project paths and baseline model/storage configuration."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF = PROJECT_ROOT / "docs" / "Tesla_Inc.pdf"
DEFAULT_DATASET = PROJECT_ROOT / "src" / "eval_dataset.json"
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"

EMBEDDING_MODEL = "voyage-4"
GENERATION_MODEL = "claude-sonnet-5"

COLLECTION_NAME = "tesla_chunks"
VECTOR_SIZE = 1_024
DISTANCE_METRIC = "cosine"
UPLOAD_BATCH_SIZE = 64
DEFAULT_TOP_K = 5
