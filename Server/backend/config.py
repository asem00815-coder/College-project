from pathlib import Path
import os

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
QDRANT_DIR = BASE_DIR / "qdrant_db"

HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "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")
HF_EMBEDDING_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
GROQ_MODEL = "llama-3.1-8b-instant"

QDRANT_URL = os.environ.get("QDRANT_URL", "https://b140b782-ead3-4b04-8d7e-257ab3032146.eu-central-1-0.aws.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6YzA1NmI2NDUtYzUyZi00N2FhLThmMjMtM2Q1NTVjOWRhZjI2In0.VsuPXAcFOc8CqRSI5YuSzK1h0b4OY3OXRrtPUqi-CyQ")


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
MAX_HISTORY = 3
MAX_NEW_TOKENS = 256

SYSTEM_PROMPT = (
    "Ты helpful ассистент. Отвечай кратко на русском языке. "
    "Если вопрос не по документам - отвечай честно из общих знаний."
)