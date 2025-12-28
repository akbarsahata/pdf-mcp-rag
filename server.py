from pathlib import Path
from typing import Any, Dict, List

import chromadb
import tantivy
from sentence_transformers import SentenceTransformer

from mcp.server.fastmcp import FastMCP

CHROMA_DIR = Path("chroma")
TANTIVY_DIR = Path("tantivy_index")
COLLECTION_NAME = "pdf_chunks"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

K0 = 60.0  # RRF constant

mcp = FastMCP("PDF Hybrid Search (MinerU + Chroma + BM25)")

# Vector index
chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma.get_or_create_collection(name=COLLECTION_NAME)
embedder = SentenceTransformer(EMBED_MODEL)

# BM25 index
bm25_index = tantivy.Index.open(str(TANTIVY_DIR))
bm25_searcher = bm25_index.searcher()


def rrf_fuse(vec_ids: List[str], bm25_ids: List[str]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for rank, _id in enumerate(vec_ids, start=1):
        scores[_id] = scores.get(_id, 0.0) + 1.0 / (K0 + rank)
    for rank, _id in enumerate(bm25_ids, start=1):
        scores[_id] = scores.get(_id, 0.0) + 1.0 / (K0 + rank)
    return scores


@mcp.tool()
def search(query: str, top_k: int = 8, vec_k: int = 12, bm25_k: int = 12) -> Dict[str, Any]:
    # Vector
    q_emb = embedder.encode([query]).tolist()[0]
    vres = collection.query(
        query_embeddings=[q_emb],
        n_results=vec_k,
        include=["documents", "metadatas", "distances"],
    )
    vec_ids = vres["ids"][0]

    # BM25
    bm25_query = bm25_index.parse_query(query)
    bres = bm25_searcher.search(bm25_query, bm25_k)
    bm25_ids: List[str] = []
    for score, addr in bres.hits:
        doc = bm25_searcher.doc(addr)
        bm25_ids.append(doc["id"][0])

    # Fuse + rank
    fused = rrf_fuse(vec_ids, bm25_ids)
    ranked_ids = sorted(fused.keys(), key=lambda x: fused[x], reverse=True)[:top_k]

    # Fetch from Chroma
    got = collection.get(ids=ranked_ids, include=["documents", "metadatas"])
    results = []
    for _id, doc, meta in zip(got["ids"], got["documents"] or [], got["metadatas"] or []):
        results.append({
            "id": _id,
            "title": f"{meta.get('source_file', 'pdf')}",
            "score": fused.get(_id, 0.0),
            "snippet": (doc[:400] + "…") if len(doc) > 400 else doc,
            "metadata": meta,
        })

    return {"results": results}


@mcp.tool()
def fetch(ids: List[str]) -> Dict[str, Any]:
    got = collection.get(ids=ids, include=["documents", "metadatas"])
    return {
        "documents": [
            {"id": _id, "content": doc, "metadata": meta}
            for _id, doc, meta in zip(got["ids"], got["documents"] or [], got["metadatas"] or [])
        ]
    }


if __name__ == "__main__":
    mcp.run()
