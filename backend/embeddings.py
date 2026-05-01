import requests
from backend.config import HF_API_TOKEN, HF_EMBEDDING_MODEL_ID

_API_URL = f"https://api-inference.huggingface.co/models/{HF_EMBEDDING_MODEL_ID}"
_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def embed_texts(texts: list[str]) -> list:
    """Эмбеддинги для списка текстов через HF API."""
    response = requests.post(
        _API_URL,
        headers=_HEADERS,
        json={"inputs": texts, "wait_for_model": True}
    )
    response.raise_for_status()
    return response.json()

def embed_query(query: str) -> list:
    """Эмбеддинг одного текста."""
    return embed_texts([query])[0]