import chromadb
from backend.config import CHROMA_DIR, TOP_K

import asyncio

_collection, _client = None, None
_lock = asyncio.Lock()

async def get_collection():
    global _client, _collection

    async with _lock:
        if _collection is None:
            _client = await asyncio.to_thread(chromadb.PersistentClient, path=CHROMA_DIR)

            _collection = _client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )

    return _collection

async def add_documents(chunks: list[dict], embeddings: list):
    collection = await get_collection()

    await asyncio.to_thread(
        collection.add,
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[{
                "source": c["source"],
                "chunk_index": c["chunk_index"]
        } for c in chunks]
    )

async def search_similar(query_embedding: list[float], top_k: int = TOP_K):
    collection = await get_collection()

    results = await asyncio.to_thread(
        collection.query,
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results

async def clear_collection():
    collection = await get_collection()

    data = await asyncio.to_thread(collection.get, include=[])
    ids = data.get("ids", [])

    if ids:
        await asyncio.to_thread(collection.delete, ids=ids)