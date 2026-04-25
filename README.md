# DeepRead AI — Document Q&A RAG API

A production-ready Retrieval-Augmented Generation (RAG) backend that accepts a document URL and a list of questions, then returns precise, context-grounded answers. Built with FastAPI, OpenAI, Astra DB (vector + keyword store), and Redis for document deduplication.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Features](#features)
- [Supported Document Types](#supported-document-types)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Environment Variables](#environment-variables)
  - [Installation](#installation)
  - [Running the Server](#running-the-server)
- [API Reference](#api-reference)
  - [POST /hackrx/run](#post-hackrxrun)
  - [GET /](#get-)
- [RAG Pipeline Deep Dive](#rag-pipeline-deep-dive)
- [Configuration & Tuning](#configuration--tuning)
- [Caching & Deduplication](#caching--deduplication)

---

## Architecture Overview

```
Document URL
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                   FastAPI Server                    │
│                                                     │
│  1. Redis check → skip if already processed         │
│  2. Download + extract text (format-specific)       │
│  3. Language detect → translate if needed           │
│  4. Chunk text (RecursiveCharacterTextSplitter)     │
│  5. Embed chunks (OpenAI text-embedding-3-small)    │
│  6. Store in Astra DB (vector + inverted index)     │
│  7. Mark doc as processed in Redis                  │
└──────────────┬──────────────────────────────────────┘
               │ On query
               ▼
┌─────────────────────────────────────────────────────┐
│               Hybrid Retrieval (Parallel)           │
│                                                     │
│  ┌─────────────────┐     ┌───────────────────────┐  │
│  │ Semantic Search │     │ Keyword Search        │  │
│  │ (ANN via Astra) │     │ (Inverted Index)      │  │
│  └────────┬────────┘     └──────────┬────────────┘  │
│           └──────────┬──────────────┘               │
│                      ▼                               │
│          Merge & Deduplicate Contexts               │
│                      ▼                               │
│   CrossEncoder Reranking (ms-marco-MiniLM-L-6-v2)  │
│          (skipped if < k_rerank candidates)          │
└──────────────┬──────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────┐
│          LLM Answer Generation (GPT-4o-mini)        │
│                                                     │
│  Chain-of-Thought prompt → single paragraph answer  │
│  Answers in the language of the question            │
└─────────────────────────────────────────────────────┘
```

---

## Features

- **Hybrid retrieval** — semantic (dense vector) + keyword (inverted index) search run in parallel
- **Adaptive reranking** — CrossEncoder reranker with weighted scoring; skipped automatically when candidate count is low
- **Multi-format ingestion** — PDF, DOCX, PPTX, XLSX/XLS, JPG/PNG, HTML/web URLs, and ZIP archives
- **Vision OCR** — images are processed via GPT-4o-mini's vision capability
- **Multilingual support** — automatic language detection and translation using `langdetect` + `deep-translator`
- **Document deduplication** — Redis-backed cache prevents reprocessing the same document
- **Async-first** — all I/O-heavy operations use `asyncio` and `aiohttp`
- **Batched embeddings** — async batching to OpenAI embeddings API (500 texts per batch)
- **Built-in minimal UI** — a Tailwind CSS frontend served at `/` for quick manual testing

---

## Supported Document Types

| Extension | Extraction Method |
|---|---|
| `.pdf` | PyMuPDF (`fitz`) |
| `.docx` | `python-docx` |
| `.pptx` | `python-pptx` |
| `.xlsx`, `.xls` | `pandas` + `openpyxl` |
| `.jpg`, `.jpeg`, `.png` | GPT-4o-mini Vision API |
| `.zip` | Extracts and routes each file recursively |
| Web URLs (no extension) | `aiohttp` scrape + `BeautifulSoup` clean |

---

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI + Uvicorn |
| Vector database | DataStax Astra DB |
| Keyword index | Astra DB (inverted index collection) |
| Cache / dedup | Redis |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) |
| LLM | OpenAI `gpt-4o-mini` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers) |
| Text splitting | LangChain `RecursiveCharacterTextSplitter` |
| Translation | `deep-translator` (GoogleTranslator) |

---

## Project Structure

```
├── main.py            # FastAPI app, lifespan management, HTTP endpoints, minimal UI
├── worker.py          # All RAG logic: extraction, indexing, retrieval, reranking, LLM generation
└── requirements.txt   # Python dependencies
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- A running Redis instance (local or cloud)
- An [Astra DB](https://astra.datastax.com) account with a database created
- An OpenAI API key with access to `text-embedding-3-small` and `gpt-4o-mini`

### Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
REDIS_URL=redis://localhost:6379
ASTRA_DB_API_ENDPOINT=https://<your-db-id>-<region>.apps.astra.datastax.com
ASTRA_DB_APPLICATION_TOKEN=AstraCS:...
```

### Installation

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server will be available at `http://localhost:8000`.

On startup, the app will:
1. Load environment variables
2. Connect to Redis and Astra DB
3. Load the CrossEncoder reranker model locally
4. Create Astra DB collections (`ragdb_semantic`, `ragdb_keyword`) if they don't exist

---

## API Reference

### `POST /hackrx/run`

Processes a document and answers a list of questions against it.

**Request Body**

```json
{
  "documents": "https://example.com/report.pdf",
  "questions": [
    "What is the main conclusion?",
    "Who are the authors?"
  ],
  "semantic_weight": 0.7,
  "keyword_weight": 0.3,
  "k_semantic": 30,
  "k_keyword": 30,
  "k_rerank": 15
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `documents` | `string` | required | URL of the document to process |
| `questions` | `string[]` | required | List of questions to answer |
| `semantic_weight` | `float` | `0.7` | Weight for semantic search hits during reranking |
| `keyword_weight` | `float` | `0.3` | Weight for keyword search hits during reranking |
| `k_semantic` | `int` | `30` | Number of candidates to retrieve from vector search |
| `k_keyword` | `int` | `30` | Number of candidates to retrieve from keyword search |
| `k_rerank` | `int` | `15` | Final number of chunks to pass to the LLM after reranking |

**Response**

```json
{
  "job_id": "report-a1b2c3d4",
  "answers": [
    "The main conclusion is ...",
    "The authors are ..."
  ]
}
```

| Field | Description |
|---|---|
| `job_id` | A stable MD5-based ID derived from the document URL |
| `answers` | One answer string per question, in the same order |

**Error Responses**

| Status | Meaning |
|---|---|
| `500` | Document download failed, text extraction failed, or LLM error |

---

### `GET /`

Returns the built-in HTML frontend (Tailwind CSS). Use this to manually test the pipeline in a browser without any additional tooling.

---

## RAG Pipeline Deep Dive

### 1. Ingestion

When a document URL is received for the first time:

1. The file extension is detected from the URL path.
2. For binary formats (PDF, DOCX, PPTX, XLSX, images), the file is downloaded to a temporary path and then deleted after processing.
3. For URLs without a recognized extension, the page is scraped with `aiohttp` and cleaned with `BeautifulSoup`.
4. If the extracted text is in a language other than English, it is translated and the translation is appended to the original text (preserving both for multilingual Q&A).

### 2. Chunking & Indexing

- Text is classified by document type (`policy`, `scientific`, `narrative`, `general`) to select chunking parameters.
- `RecursiveCharacterTextSplitter` splits text into chunks of ~1000 characters with 100-character overlap.
- Each chunk is embedded using `text-embedding-3-small` (batched, async).
- Chunks are stored in Astra DB's vector collection with their `document_id`.
- An inverted index of NLTK-filtered keyword tokens → chunk IDs is built and stored in a separate Astra DB collection (in batches of 200 to avoid timeouts).
- The `document_id` is added to a Redis set to mark the document as processed.

### 3. Retrieval

For each incoming question:

- **Semantic search** — the query is embedded and an ANN (approximate nearest-neighbor) search is run against the vector collection, filtered by `document_id`.
- **Keyword search** — the query is tokenized (stopwords removed), a lookup is performed against the inverted index, and chunks are ranked by token hit frequency.
- Both searches run concurrently via `asyncio.gather`.
- Results are merged and deduplicated, tagged with their source (`semantic`, `keyword`, or `both`).

### 4. Reranking

If the combined candidate count is ≥ `k_rerank`:

- All query–chunk pairs are scored by the CrossEncoder in a single batched call.
- Scores are weighted: `both` sources receive `semantic_weight + keyword_weight`, `semantic` receives `semantic_weight`, `keyword` receives `keyword_weight`.
- The top `k_rerank` chunks are selected.

If fewer than `k_rerank` candidates exist, the reranker is skipped to avoid overhead.

### 5. Answer Generation

The top-ranked chunks are assembled into a context block and sent to `gpt-4o-mini` with a structured Chain-of-Thought prompt. The model is instructed to:
- Answer strictly from the provided context
- Respond in the language of the question
- Return a single clean paragraph as the final answer

All questions for a given request are answered in parallel with `asyncio.gather`.

---


## Caching & Deduplication

Documents are tracked in a Redis set under the key `"v2"` (the value of `PROCESSED_DOCS_KEY`). A document is identified by a stable MD5 hash of its normalized URL path + query string. If the document has been processed before, the entire ingestion pipeline is skipped and the system goes directly to retrieval.

To force reprocessing of a document (e.g. after a source update), delete its `document_id` from the Redis set:

```bash
redis-cli SREM v2 <document_id>
```