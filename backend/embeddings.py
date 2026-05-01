from sentence_transformers import SentenceTransformer
from backend.config import HF_EMBEDDING_MODEL_ID

model = SentenceTransformer(HF_EMBEDDING_MODEL_ID)

def embed_texts(texts: list[str]) -> list:
    embeddings = model.encode(texts)
    return embeddings.tolist()

def embed_query(query: str) -> list:
    return embed_texts([query])[0]