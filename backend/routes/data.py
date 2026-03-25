from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from embeddings import embed_texts
from database import add_documents, get_collection

from fastapi import APIRouter, HTTPException

from pathlib import Path
import uuid


router = APIRouter()

def chunk_text(text: str, source: str) -> list[dict]:
    chunks = []
    start = 0
    index = 0
    
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk,
                "source": source,
                "chunk_index": index
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP
        index += 1
    return chunks

@router.post("/load")
def load_documents():
    if not DATA_DIR.exists():
        raise HTTPException(status_code=404, detail="Папка data не найдена")

    txt_files = list(DATA_DIR.glob("*.txt"))
    if not txt_files:
        raise HTTPException(status_code=404, detail="Нет .txt файлов в папке data")

    all_chunks = []
    for filepath in txt_files:
        text = filepath.read_text(encoding="utf-8")
        chunks = chunk_text(text, source=filepath.name)
        all_chunks.extend(chunks)

    embeddings = embed_texts([c["text"] for c in all_chunks])
    add_documents(all_chunks, embeddings)

    return {"status": "ok", "loaded_files": txt_files, "total_chunks": len(all_chunks)}

@router.get("/status")
def get_status():
    collection = get_collection()
    return {"total_chunks": collection.count()}