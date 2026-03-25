from backend.embeddings import embed_query
from backend.database import search_similar
from backend.config import TOP_K

def retrieve_context(query: str) -> tuple[str, list[str]]:
    query_embedding = embed_query(query)
    results = search_similar(query_embedding, top_k=TOP_K)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    context = "\n\n".join(documents)
    sources = list(set([m["source"] for m in metadatas]))

    return context, sources

def build_prompt(query: str, context: str) -> str:
    return f"""Ты ассистент по историческим фактам удельных центров россии. Отвечай на русском языке на основе предоставленного контекста.
Если ответа нет в контексте скажи об этом честно.

Контекст:
{context}

Вопрос: {query}

Ответ:"""