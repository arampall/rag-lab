# RAG Lab

An incremental, evaluation-driven project for learning how to build a
retrieval-augmented generation (RAG) system.

The project intentionally starts with understandable components and introduces
new techniques only after the current baseline has been inspected and tested.

## Current pipeline

```text
Tesla PDF
  -> page-by-page extraction with pypdf
  -> conservative whitespace cleaning
  -> 600-token chunks with 100-token overlap
  -> Voyage document embeddings
  -> local Qdrant vector index
```

Single-query retrieval inspection is implemented. Full retrieval evaluation and
answer generation have not been added yet.

## Project structure

```text
.
├── docs/
│   └── Tesla_Inc.pdf
├── src/
│   ├── extract.py
│   ├── chunk.py
│   ├── embed.py
│   ├── evaluate.py
│   ├── eval_dataset.json
│   ├── index_preflight.py
│   ├── index_qdrant.py
│   ├── retrieve.py
│   └── main.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Create the local environment file:

```bash
cp .env.example .env
```

Add a newly generated Voyage API key to `.env`:

```text
VOYAGE_API_KEY=your_voyage_api_key
```

Never commit `.env` or hard-code credentials. In production, inject secrets
through the deployment platform or a secret manager.

## Inspect PDF extraction

Print the default representative pages as raw extracted text:

```bash
python src/extract.py
```

Compare raw and conservatively cleaned page text:

```bash
python src/extract.py --pages 3 --max-chars 0
python src/extract.py --pages 3 --max-chars 0 --clean
```

Cleaning currently normalizes safe whitespace artifacts. It deliberately does
not guess paragraph boundaries, reorder content, or identify headings.

## Build chunks and inspect embeddings

Run the connected pipeline:

```bash
python src/main.py
```

The chunking baseline uses:

```text
embedding model: voyage-4
chunk size:     600 tokens
chunk overlap:  100 tokens
```

Chunks stay within individual PDF pages so every chunk has an unambiguous page
number. Section and subsection metadata remain unset until heading detection is
implemented and validated.

Document chunks must be embedded with `input_type="document"`. User questions
will later be embedded with `input_type="query"`.

## Validate the retrieval evaluation dataset

Run the offline dataset and source-grounding checks:

```bash
python src/evaluate.py
```

The checker validates the record schema, stable unique IDs, page references,
and expected source phrases. It reads the local PDF but does not create
embeddings or make API calls.

The initial retrieval metric is page `Hit@5`: a question passes when at least
one of its five highest-ranked chunks comes from one of its expected pages.
Retrieved chunk text and similarity scores will also be printed for diagnosing
why individual questions miss.

## Preview Qdrant indexing

Inspect the complete indexing scope without making API calls:

```bash
python src/index_preflight.py
```

The preflight reports how many cleaned chunk texts and tokens would be sent to
Voyage, checks chunk IDs and metadata, and displays the planned Qdrant vector
and payload configuration. It does not create embeddings or a collection.

Preview the guarded indexing command:

```bash
python src/index_qdrant.py
```

After inspecting its scope, explicitly execute document embedding and indexing:

```bash
python src/index_qdrant.py --execute
```

Execution stores the `tesla_chunks` collection under `data/qdrant/`. It creates
the collection only when absent, rejects incompatible existing configuration,
upserts deterministic points in batches, and verifies the final point count and
a sample payload. The local Qdrant data is ignored by Git.

## Inspect one retrieval query

Preview the default evaluation query without making an API call:

```bash
python src/retrieve.py
```

Embed that one question as a Voyage query and inspect Qdrant's top five chunks:

```bash
python src/retrieve.py --execute
```

Use `--example-id` to select another stable ID from `eval_dataset.json`. Output
includes rank, similarity score, page, chunk ID, expected-phrase matches, and a
short text preview so retrieval failures remain inspectable.

## Learning principle

Retrieval and generation are evaluated separately:

- If the relevant chunk is not retrieved, debug retrieval.
- If the relevant chunk is retrieved but the answer is poor, debug generation.

The next stages will evaluate retrieval across all eight questions and then add
grounded answer generation only after retrieval failures are understood.
