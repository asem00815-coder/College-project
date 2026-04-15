from fastapi import APIRouter
from backend.models.documents import ChatRequest, ChatResponse
from backend.rag import retrieve_context, build_prompt
from transformers import pipeline
from backend.config import LLM_MODEL, MAX_HISTORY, MAX_NEW_TOKENS
import torch
import asyncio

router = APIRouter()

_llm = None

async def get_llm():
    global _llm

    if _llm is None:
        _llm = pipeline(
            task="text-generation",
            model=LLM_MODEL,
            dtype=torch.bfloat16,
            device_map="cpu",
            max_new_tokens=MAX_NEW_TOKENS
        )
    
    return _llm

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    context, sources = await retrieve_context(request.message)
    prompt = build_prompt(request.message, context, request.history[-MAX_HISTORY:])

    llm = await get_llm()
    output = await asyncio.to_thread(
        llm,
        prompt,
        do_sample=False
    )
    
    answer = output[0]["generated_text"][len(prompt):].strip()

    return ChatResponse(answer=answer, sources=sources)