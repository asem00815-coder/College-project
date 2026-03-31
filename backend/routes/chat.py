from fastapi import APIRouter
from backend.models.documents import ChatRequest, ChatResponse
from backend.rag import retrieve_context, build_prompt
from transformers import pipeline
from backend.config import LLM_MODEL
import torch

router = APIRouter()

_llm = None

def get_llm():
    global _llm

    if _llm is None:
        _llm = pipeline(
            "text-generation",
            model=LLM_MODEL,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
            max_new_tokens=256
        )
    
    return _llm

@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):
    context, sources = retrieve_context(request.message)
    prompt = build_prompt(request.message, context, request.history)

    llm = get_llm()
    output = llm(prompt, do_sample=False)
    
    answer = output[0]["generated_text"][len(prompt):].strip()

    return ChatResponse(answer=answer, sources=sources)