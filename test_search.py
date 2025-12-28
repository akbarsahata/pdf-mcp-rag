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

# test fetch
vres_fetched = collection.get(ids=vres["ids"][0], include=["metadatas"])
print("Vector fetched metadatas:", vres_fetched["metadatas"])
bm_fetched = []
for bid in bm_ids:
    doc = searcher.doc(tantivy.DocAddress(0, int(bid)))
    bm_fetched.append(doc)
print("BM25 fetched metadatas:", [doc["metadata"][0] for doc in bm_fetched])
assert vres_fetched["metadatas"] == vres["metadatas"]
assert [doc["metadata"][0] for doc in bm_fetched] == bm_ids
print("All tests passed.")
