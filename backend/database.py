import qdrant_client
from qdrant_client.http import models as rest
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from backend.config import QDRANT_DIR, TOP_K

_client = None
_vector_store = None
_embedding_dim = None
COLLECTION_NAME = "documents"

def _ensure_collection_exists(dim: int):
    global _client
    if not _client.collection_exists(COLLECTION_NAME):
        _client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=rest.VectorParams(
                size=dim,
                distance=rest.Distance.COSINE
            )
        )

def get_vector_store() -> QdrantVectorStore:
    global _client, _vector_store
    if _vector_store is None:
        _client = qdrant_client.QdrantClient(path=str(QDRANT_DIR))
        _vector_store = QdrantVectorStore(
            client=_client,
            collection_name=COLLECTION_NAME
        )
    return _vector_store

def add_documents(chunks: list[dict], embeddings: list):
    global _embedding_dim
    store = get_vector_store()

    if _embedding_dim is None and embeddings:
        _embedding_dim = len(embeddings[0])
        _ensure_collection_exists(_embedding_dim)

    nodes = [
        TextNode(
            id_=c["id"],
            text=c["text"],
            metadata={"source": c["source"], "chunk_index": c["chunk_index"]},
            embedding=emb
        )
        for c, emb in zip(chunks, embeddings)
    ]
    store.add(nodes)

def search_similar(query_embedding: list, top_k: int = TOP_K):
    store = get_vector_store()
    if not _client.collection_exists(COLLECTION_NAME):
        return {"documents": [[]], "metadatas": [[]]}

    query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=top_k,
    )
    result = store.query(query)

    docs = []
    metas = []
    if result.nodes:
        for node in result.nodes:
            docs.append(node.text)
            metas.append(node.metadata)

    return {
        "documents": [docs],
        "metadatas": [metas],
    }

def clear_collection():
    global _client
    get_vector_store()
    if _client.collection_exists(COLLECTION_NAME):
        _client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=rest.Filter(),
        )

def get_document_count() -> int:
    store = get_vector_store()
    try:
        return store.client.count(COLLECTION_NAME).count
    except Exception:
        return 0