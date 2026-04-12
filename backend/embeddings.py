from sentence_transformers import SentenceTransformer
from backend.config import EMBEDDING_MODEL
import asyncio

_model = None
_lock = asyncio.Lock()

async def get_embedding_model():
    global _model
    
    if _model is None:
        async with _lock:
            _model = await asyncio.to_thread(SentenceTransformer, EMBEDDING_MODEL)

    return _model

async def embed_texts(texts: list[str]) -> list:
    model = await get_embedding_model()

    embeddings = await asyncio.to_thread(model.encode, texts, show_progress_bar=True, normalize_embeddings=True)

    return embeddings.tolist()

async def embed_query(query: str) -> list:
    model = await get_embedding_model()
    embeddings = await asyncio.to_thread(model.encode, [query], normalize_embeddings=True)
    return embeddings[0].tolist()