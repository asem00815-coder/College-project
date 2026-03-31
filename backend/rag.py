from backend.embeddings import embed_query
from backend.database import search_similar
from backend.config import TOP_K, MAX_HISTORY

def retrieve_context(query: str) -> tuple[str, list[str]]:
    query_embedding = embed_query(query)
    results = search_similar(query_embedding, top_k=TOP_K)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    context = "\n\n".join(documents)
    sources = list(set([m["source"] for m in metadatas]))

    return context, sources

def build_prompt(query: str, context: str, history: list = []) -> str:
    history_text = ""
    for msg in history[-MAX_HISTORY:]:
        role = "Пользователь" if msg["role"] == "user" else "Ассистент"
        history_text += f"{role}: {msg['content']}\n"

    return f"""<|im_start|>system
Ты helpful ассистент. Отвечай кратко на русском языке.
Если вопрос не по документам — отвечай честно из общих знаний.
Контекст из документов: {context if context else 'нет'}
<|im_end|>
<|im_start|>user
{history_text}Пользователь: {query}
<|im_end|>
<|im_start|>assistant
"""