import chromadb
import tantivy
from sentence_transformers import SentenceTransformer

# Vector
chroma = chromadb.PersistentClient(path="chroma")
collection = chroma.get_or_create_collection("pdf_chunks")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# BM25
bm25_index = tantivy.Index.open("tantivy_index")
searcher = bm25_index.searcher()

q = "introduction"
# vector
q_emb = embedder.encode([q]).tolist()[0]
vres = collection.query(query_embeddings=[q_emb], n_results=3, include=["metadatas"])
print("Vector top ids:", vres["ids"][0])

# bm25
bq = bm25_index.parse_query(q)
bres = searcher.search(bq, 3)
bm_ids = []
for score, addr in bres.hits:
    doc = searcher.doc(addr)
    bm_ids.append(doc["id"][0])
print("BM25 top ids:", bm_ids)
