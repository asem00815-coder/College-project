import chromadb
from backend.config import CHROMA_DIR

_collection, _client = None, None

def get_collection():
    global _client, _collection

    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)

        _collection = _client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

    return _collection

def add_documents(chunks: list[dict], embeddings: list):
    collection = get_collection()

    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks]
    )

def search_similar(query_embedding: list, top_k: int = 3):
    collection = get_collection()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results