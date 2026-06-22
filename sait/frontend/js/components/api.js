const API_URL = "https://asem00815-college-rag.hf.space";

async function sendMessage(message, history) {
    const response = await fetch(`${API_URL}/chat/`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            message: message,
            history: history
        })
    });

    if (!response.ok) {
        throw new Error("Ошибка сервера");
    }

    return await response.json();
}

async function loadDocuments() {
    const response = await fetch(`${API_URL}/data/load`, {method: "POST"});
    return await response.json();
}

async function getStatus() {
    const response = await fetch(`${API_URL}/data/status`);
    return await response.json();
}