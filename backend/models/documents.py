from pydantic import BaseModel
from typing import Optional

class DocumentChunk(BaseModel):
    id: str
    text: str
    source: str
    chunk_index: int

class ChatRequest(BaseModel):
    message: str
    history: Optional[list] = []

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]