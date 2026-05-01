from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
QDRANT_DIR = BASE_DIR / "qdrant_db"

HF_API_TOKEN = "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"
HF_EMBEDDING_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

GROQ_API_KEY = "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"
GROQ_MODEL = "llama-3.1-8b-instant"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
MAX_HISTORY = 3
MAX_NEW_TOKENS = 256

SYSTEM_PROMPT = (
    "Ты helpful ассистент. Отвечай кратко на русском языке. "
    "Если вопрос не по документам - отвечай честно из общих знаний."
)