from backend.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from backend.embeddings import embed_texts
from backend.database import add_documents, get_collection, clear_collection

from fastapi import APIRouter, HTTPException

from pathlib import Path
import uuid
import asyncio

router = APIRouter()

_load_semaphore = asyncio.Semaphore(1)

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

async def process_documents():
    collection = get_collection()
    if collection.count() > 0:
        raise HTTPException(status_code=400, detail="Документы уже загружены.")

    if not DATA_DIR.exists():
        raise HTTPException(status_code=404, detail="Папка data не найдена")

    txt_files = list(DATA_DIR.glob("**/*.txt"))
    if not txt_files:
        raise HTTPException(status_code=404, detail="Нет .txt файлов в папке data")

    all_chunks = []
    for filepath in txt_files:
        text = filepath.read_text(encoding="utf-8")
        chunks = chunk_text(text, source=filepath.name)
        all_chunks.extend(chunks)

    embeddings = embed_texts([c["text"] for c in all_chunks])
    add_documents(all_chunks, embeddings)

    return {"status": "ok",
            "loaded_files": [f.name for f in txt_files],
            "total_chunks": len(all_chunks)
    }

@router.post("/load")
async def load_documents():
    async with _load_semaphore:
        try:
            return await asyncio.to_thread(process_documents)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Ошибка при загрузке документов: {str(e)}")

@router.get("/status")
async def get_status():
    collection = await asyncio.to_thread(get_collection)
    return {"total_chunks": collection.count()}

@router.delete("/clear")
async def clear_documents():
    await asyncio.to_thread(clear_collection)
    return {"status": "ok", "message": "База очищена"}