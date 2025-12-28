# PDF-MCP-RAG

A **Model Context Protocol (MCP) server** that enables Claude and other AI systems to perform **hybrid semantic + lexical search** over PDF documents using **MinerU extraction**, **ChromaDB vector embeddings**, and **Tantivy BM25 full-text search**.

## Overview

This project provides a production-ready system for intelligent PDF document retrieval combining:

- **MinerU**: Advanced PDF extraction with layout awareness (markdown + structured JSON)
- **ChromaDB**: Vector database for semantic (dense) search
- **Sentence Transformers**: Embedding model for semantic understanding
- **Tantivy**: BM25 full-text search engine
- **MCP Framework**: Seamless integration with Claude, VS Code, and compatible tools

The system implements **Reciprocal Rank Fusion (RRF)** to blend vector and BM25 results, providing more robust retrieval than either method alone.

## Features

✅ **Automated PDF Processing**: Use MinerU to extract structured text and metadata from PDFs  
✅ **Dual-Index Search**: Semantic + lexical ranking with RRF fusion  
✅ **MCP Integration**: Expose search as tools for Claude Desktop, VS Code, and other MCP clients  
✅ **Chunking & Caching**: Smart text chunking with configurable overlap; caches extracted content  
✅ **JSON & Markdown Support**: Prefers MinerU's structured JSON output, falls back to markdown  
✅ **Batch Indexing**: Handles large document collections with ChromaDB batch size management

## Example of Use Case

I used this system to build a document retrieval tool for educational research papers on misconception detection in AI-driven learning systems. The dual-index search allowed me to find relevant papers even when terminology varied across disciplines. Below are some of the key papers I explored:

- [Demirezen et al. (2023) – Physics misconceptions via Transformers](https://link.springer.com/10.1007/s00521-023-08414-2)
- [Otero et al. (2025) – Algebra misconception benchmarks with LLMs](https://link.springer.com/10.1007/s44217-025-00742-w)
- [Kökver et al. (2024) – NLP-based environmental science misconceptions](https://link.springer.com/10.1007/s10639-024-12919-1)
- [Fischer et al. (2023) – Programming education feedback systems](https://dl.acm.org/doi/10.1145/3629296.3629297)

I connect the MCP server to VS Code using the MCP extension, allowing me to query the document collection directly from my editor while writing research summaries. As for AI models, I primarily use GPT-5.2.

The [`from-markdown.md`](from-markdown.md) and [`from-json.md`](from-json.md) file contain synthesized literature review based on these papers, formatted in IEEE style. The filenames indicate whether the content was extracted from MinerU's markdown or JSON output.

See the files for results and comparisons of extraction quality!

## Project Structure

```
pdf-mcp-rag/
├── ingest_mineru.py       # PDF extraction, chunking, and dual-index ingestion
├── server.py              # MCP server with search() and fetch() tools
├── test_search.py         # Standalone test script for search functionality
├── papers.md              # Literature review on misconception detection (IEEE-style)
├── LICENSE.md             # AGPL-3.0 licensing (due to MinerU)
├── README.md              # This file
├── package.json           # Node.js dependencies (for MCP Inspector)
├── data/
│   ├── pdfs/              # Place your PDF files here
│   └── mineru_out/        # MinerU extraction output (cached, gitignored)
├── chroma/                # ChromaDB vector store (gitignored)
└── tantivy_index/         # Tantivy BM25 index (gitignored)
```

## Installation

### Prerequisites

- **Python 3.10+**
- **Pip** (with `wheel` and `setuptools`)

### Setup

1. **Clone and create virtual environment:**
   ```bash
   git clone <repo-url>
   cd pdf-mcp-rag
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Upgrade pip, wheel, and setuptools:**
   ```bash
   pip install -U pip wheel setuptools
   ```

3. **Install core dependencies:**
   ```bash
   pip install "pymupdf>=1.24" "chromadb>=0.5" "sentence-transformers>=3" "tantivy>=0.22" "fastapi>=0.110" "uvicorn>=0.27" "pydantic>=2"
   ```

4. **Install MCP framework:**
   ```bash
   pip install "mcp>=1.0"
   ```

5. **Install MinerU (for PDF extraction):**
   ```bash
   pip install uv
   uv pip install -U "mineru[core]"
   ```

6. **Place PDFs in `data/pdfs/`:**
   ```bash
   mkdir -p data/pdfs
   cp your-documents.pdf data/pdfs/
   ```

**Quick Install (all-in-one):**
```bash
pip install -U pip wheel setuptools
pip install -r requirements.txt
pip install uv
uv pip install -U "mineru[core]"
```

## Usage

### 1. Ingest PDFs

Extract PDFs using MinerU and build dual indexes (ChromaDB + Tantivy):

```bash
python ingest_mineru.py
```

**Output:**
- `data/mineru_out/` – MinerU extraction results (markdown + JSON)
- `chroma/` – Vector database with embeddings
- `tantivy_index/` – BM25 full-text index

The script:
- Runs MinerU on each PDF (if not already extracted)
- Extracts text from JSON or markdown
- Chunks text (default: 1400 chars, 200 char overlap)
- Embeds chunks using Sentence Transformers
- Indexes in ChromaDB (with batching to handle large collections)
- Indexes in Tantivy for BM25 ranking

### 2. Test Search (Standalone)

```bash
python test_search.py
```

Runs a quick vector + BM25 search for a test query and prints top results.

### 3. Start MCP Server

```bash
python server.py
```

Launches the FastMCP server on **stdio** (default transport for MCP clients). Use with:
- **Claude Desktop**: Configure in `claude_desktop_config.json`
- **MCP Inspector**: `npx @modelcontextprotocol/inspector python server.py`
- **VS Code**: Install MCP extension and add server config

### Available Tools

#### `search(query: str, top_k: int = 8, vec_k: int = 12, bm25_k: int = 12) -> Dict`

Hybrid search combining vector + BM25 results.

**Parameters:**
- `query`: Search text
- `top_k`: Final results to return (default: 8)
- `vec_k`: Number of vector results to fetch (default: 12)
- `bm25_k`: Number of BM25 results to fetch (default: 12)

**Returns:**
```json
{
  "results": [
    {
      "id": "chunk-uuid",
      "title": "source_file.pdf",
      "score": 0.45,
      "snippet": "text preview (first 400 chars)...",
      "metadata": {"page": 5, "section": "Introduction", "chunk_index": 2}
    }
  ]
}
```

#### `fetch(ids: List[str]) -> Dict`

Retrieve full document content by chunk IDs.

**Parameters:**
- `ids`: List of chunk UUIDs

**Returns:**
```json
{
  "documents": [
    {
      "id": "chunk-uuid",
      "content": "full text of chunk",
      "metadata": {...}
    }
  ]
}
```

## Configuration

Edit constants in `ingest_mineru.py` and `server.py`:

```python
CHUNK_CHARS = 1400       # Characters per chunk
CHUNK_OVERLAP = 200      # Overlap between chunks
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
K0 = 60.0                # RRF constant (higher = less ranking fusion)
```

## Architecture

### Ingestion Pipeline

```
PDF → MinerU → JSON/Markdown → Text Extraction → Chunking → 
  ├─ Embedding (Sentence Transformers) → ChromaDB (Vector Index)
  └─ BM25 Indexing → Tantivy (Full-Text Index)
```

### Search Pipeline

```
Query → 
  ├─ Vector Search (ChromaDB) → Top K vector results
  ├─ BM25 Search (Tantivy) → Top K lexical results
  └─ RRF Fusion → Ranked Combined Results
```

**Reciprocal Rank Fusion (RRF):**
```
score(d) = Σ 1/(K + rank(d))  for d in results
```
This balances semantic and lexical ranking without requiring score calibration.

## Hybrid Search Advantages

| Approach | Strength | Weakness |
|----------|----------|----------|
| **Vector (Semantic)** | Captures meaning, paraphrases | Vulnerable to out-of-vocabulary terms |
| **BM25 (Lexical)** | Exact matches, domain terms | Misses semantic relationships |
| **RRF Fusion** | Best of both worlds | Slight computational overhead |

## Development

### Running Tests

```bash
python test_search.py
```

### Debugging

Set `RUST_BACKTRACE=1` for Tantivy error details:
```bash
RUST_BACKTRACE=1 python ingest_mineru.py
```

### Modifying the Embedding Model

Change `EMBED_MODEL` in both scripts:
```python
EMBED_MODEL = "all-mpnet-base-v2"  # Larger, slower, more accurate
```

Then re-run `ingest_mineru.py` to rebuild indexes.

## Additional Resources

| `mineru-pdf` | PDF extraction | AGPL-3.0 |
| `fastmcp` | MCP framework | MIT |
| `chromadb` | Vector database | Apache 2.0 |
| `tantivy` | BM25 search engine | MIT |
| `sentence-transformers` | Embedding models | Apache 2.0 |
