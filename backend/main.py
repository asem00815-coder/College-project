from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import chat, data

app = FastAPI(title="RAG Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data.router, prefix="/data", tags=["data"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])