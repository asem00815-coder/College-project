from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3