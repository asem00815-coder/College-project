from fastapi import APIRouter, HTTPException
from backend.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from backend.embeddings import embed_texts
from backend.database import add_documents, get_document_count, clear_collection
import uuid
import fitz

router = APIRouter()

def extract_text_from_pdf(filepath):
    text = ""
    with fitz.open(filepath) as doc:
        for page in doc:
            text += page.get_text()
    return text

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
    if get_document_count() > 0:
        raise HTTPException(status_code=400, detail="Документы уже загружены.")

    if not DATA_DIR.exists():
        raise HTTPException(status_code=404, detail="Папка data не найдена")

    files = list(DATA_DIR.glob("**/*.txt")) + list(DATA_DIR.glob("**/*.pdf"))
    
    if not files:
        raise HTTPException(status_code=404, detail="Нет поддерживаемых файлов (.txt, .pdf) в папке data")

    all_chunks = []
    for filepath in files:
        try:
            if filepath.suffix.lower() == ".txt":
                text = filepath.read_text(encoding="utf-8")
            elif filepath.suffix.lower() == ".pdf":
                text = extract_text_from_pdf(filepath)
            else:
                continue
            
            if not text.strip():
                continue

            chunks = chunk_text(text, source=filepath.name)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Ошибка при чтении файла {filepath.name}: {e}")
            continue

    if not all_chunks:
        raise HTTPException(status_code=400, detail="Не удалось извлечь текст из файлов")

    embeddings = embed_texts([c["text"] for c in all_chunks])
    add_documents(all_chunks, embeddings)

    return {"status": "ok", "chunks_loaded": len(all_chunks)}

@router.get("/status")
def get_status():
    return {"total_chunks": get_document_count()}

@router.delete("/clear")
def clear_documents():
    clear_collection()
    return {"status": "ok", "message": "База очищена"}