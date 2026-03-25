from embeddings import embed_query
from database import search_similar
from config import TOP_K

def retrieve_context(query: str) -> tuple[str, list[str]]:
    query_embedding = embed_query(query)
    results = search_similar(query_embedding, top_k=TOP_K)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    context = "\n\n".join(documents)
    sources = list(set([m["source"] for m in metadatas]))

    return context, sources

def build_prompt(query: str, context: str) -> str:
    return f"""<|im_start|>system
Ты помощник по истории. Отвечай ТОЛЬКО на основе текста ниже. 
Если информации нет, ответь: "В предоставленных документах нет ответа".
КОНТЕКСТ:
{context}<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""