# PDF-MCP-RAG

A **Model Context Protocol (MCP) server** that enables Claude and other AI systems to perform **hybrid semantic + lexical search** over PDF documents using **MinerU extraction**, **ChromaDB vector embeddings**, and **Tantivy BM25 full-text search**.

## Overview

PDF-MCP-RAG is a production-ready **hybrid PDF retrieval system** designed for semantic + lexical search. It combines:

- **MinerU**: Advanced PDF extraction with layout awareness (markdown + structured JSON)
- **ChromaDB**: Vector database for semantic (dense) search via Sentence Transformers
- **Tantivy**: BM25 full-text search engine for exact keyword matching
- **Reciprocal Rank Fusion (RRF)**: Intelligent fusion of vector and lexical results
- **MCP Framework**: Expose search as tools for Claude, VS Code, and compatible applications

Hybrid search outperforms either method alone: semantic search captures meaning and paraphrases, while BM25 captures exact domain terms. RRF combines them without requiring score calibration.

## Features

✅ **Hybrid Search**: Combine semantic (vector) and lexical (BM25) retrieval via Reciprocal Rank Fusion  
✅ **Advanced PDF Extraction**: MinerU with layout awareness; supports both markdown and structured JSON output  
✅ **MCP Integration**: Expose search tools to Claude Desktop, VS Code, and MCP-compatible applications  
✅ **Smart Chunking**: Configurable text chunking with overlap; automatic caching of extracted content  
✅ **Batch Scaling**: Efficiently handle large document collections with ChromaDB batch processing  
✅ **Dual Indexing**: ChromaDB for semantic search + Tantivy BM25 for full-text search

## Use Case Example: Educational Research Retrieval

This system powers intelligent literature search for educational AI research. A typical workflow:

1. **Ingest**: Extract 20+ papers on misconception detection, learning analytics, and grading systems
2. **Search**: Query "LLM reliability in student assessment" → hybrid search finds papers using varied terminology
3. **Fetch**: Retrieve full passages for synthesis into research summaries
4. **Integrate**: Connect MCP server to VS Code; use Claude to generate literature reviews on-demand

**Key papers indexed:**
- Otero et al. (2025) – Algebra misconception benchmarks [source](https://link.springer.com/10.1007/s44217-025-00742-w)
- Smart et al. (2024) – LLM error pattern alignment with students [source](https://link.springer.com/10.1007/978-3-031-60609-0_21)
- Qiu et al. (2024) – RAG-based grading systems [source](https://ieeexplore.ieee.org/document/10825385/)
- Grévisse (2024) – LLM consistency in educational assessment [source](https://bmcmededuc.biomedcentral.com/articles/10.1186/s12909-024-06026-5)
- and 15+ more...

The hybrid search excels here because educational papers use inconsistent terminology ("misconception diagnosis" vs. "error detection" vs. "learning gap identification"). Semantic search captures the intent; BM25 ensures no important citations are missed.

**Extensible to any domain**: Legal document discovery, medical literature synthesis, technical documentation search, knowledge base Q&A. See `from-markdown.md` and `from-json.md` for more examples.

## Project Structure

```
pdf-mcp-rag/
├── Core RAG System
│   ├── ingest_mineru.py       # PDF extraction, chunking, dual-index ingestion
│   ├── server.py              # MCP server with search() and fetch() tools
│   └── test_search.py         # Standalone test script for retrieval
├── Data & Indexes
│   ├── data/pdfs/             # Input: Place PDF files here
│   ├── data/mineru_out/       # MinerU extraction output (cached)
│   ├── chroma/                # ChromaDB vector embeddings (gitignored)
│   └── tantivy_index/         # Tantivy BM25 index (gitignored)
├── Examples & Documentation
│   ├── from-markdown.md       # Literature synthesis from MinerU markdown
│   ├── from-json.md           # Literature synthesis from MinerU JSON
│   ├── judgements.md          # Annotation notes
│   └── README.md              # This file
└── Config & Dependencies
    ├── requirements.txt       # Python dependencies
    ├── package.json           # Node.js deps (MCP Inspector)
    └── LICENSE.md             # AGPL-3.0 (due to MinerU)
```

**Note:** LaTeX and BibTeX files (*.tex, *.bib) are gitignored to keep the repo lean. Keep your paper drafts and Zotero-exported BibTeX locally.

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

If you changed the ingestion/chunking logic and want a clean rebuild (recommended to avoid mixing old and new chunks):

```bash
python ingest_mineru.py --reset --yes
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
