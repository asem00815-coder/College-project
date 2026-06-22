from groq import Groq
from backend.config import GROQ_API_KEY, GROQ_MODEL, TOP_K, MAX_HISTORY, SYSTEM_PROMPT
from backend.embeddings import embed_query
from backend.database import search_similar

_groq_client = Groq(api_key=GROQ_API_KEY)

def retrieve_context(query: str) -> tuple[str, list[str]]:
    query_embedding = embed_query(query)
    results = search_similar(query_embedding, top_k=TOP_K)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    context = "\n\n".join(documents)
    sources = list(set(m["source"] for m in metadatas))
    return context, sources

def build_messages(query: str, context: str, history: list[dict]) -> list[dict]:
    system_content = f"{SYSTEM_PROMPT}\n\nКонтекст из документов:\n{context}" if context else f"{SYSTEM_PROMPT}\n\nКонтекст: нет"
    messages = [{"role": "system", "content": system_content}]

    for msg in history[-MAX_HISTORY:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": query})
    return messages

def generate_answer(query: str, context: str, history: list[dict] = []) -> dict:
    messages = build_messages(query, context, history)
    completion = _groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=1024,
        top_p=0.95,
        stream=False,
    )
    answer = completion.choices[0].message.content
    return {"answer": answer, "model": completion.model}