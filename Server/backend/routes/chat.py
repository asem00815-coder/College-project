from fastapi import APIRouter
from backend.models.documents import ChatRequest, ChatResponse
from backend.rag import retrieve_context, generate_answer
from backend.config import MAX_HISTORY

router = APIRouter()

@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):
    context, sources = retrieve_context(request.message)
    result = generate_answer(
        query=request.message,
        context=context,
        history=request.history[-MAX_HISTORY:]
    )
    return ChatResponse(answer=result["answer"], sources=sources)