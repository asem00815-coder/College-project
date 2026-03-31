let history = [];

const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send-btn");
const statusEl = document.getElementById("status");

// Проверяем статус при загрузке
async function checkStatus() {
    try {
        const data = await getStatus();
        if (data.total_chunks > 0) {
            statusEl.textContent = `🟢 ${data.total_chunks} чанков загружено`;
        } else {
            statusEl.textContent = "🔴 Документы не загружены";
        }
    } catch {
        statusEl.textContent = "🔴 Сервер недоступен";
    }
}

// Добавить сообщение в чат
function addMessage(text, role, sources = []) {
    const div = document.createElement("div");
    div.className = `message ${role}`;

    let sourcesHtml = "";
    if (sources.length > 0) {
        sourcesHtml = `<div class="sources">📄 ${sources.join(", ")}</div>`;
    }

    div.innerHTML = `
        <div class="bubble">${text}</div>
        ${sourcesHtml}
    `;

    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

// Показать индикатор загрузки
function addLoading() {
    const div = document.createElement("div");
    div.className = "message bot loading";
    div.innerHTML = `<div class="bubble">...</div>`;
    div.id = "loading";
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function removeLoading() {
    const el = document.getElementById("loading");
    if (el) el.remove();
}

// Отправить сообщение
async function send() {
    const text = inputEl.value.trim();
    if (!text) return;

    inputEl.value = "";
    sendBtn.disabled = true;

    addMessage(text, "user");
    history.push({ role: "user", content: text });

    addLoading();

    try {
        const data = await sendMessage(text, history);
        removeLoading();
        addMessage(data.answer, "bot", data.sources);
        history.push({ role: "assistant", content: data.answer });
    } catch (e) {
        removeLoading();
        addMessage("Ошибка соединения с сервером.", "bot");
    }

    sendBtn.disabled = false;
    inputEl.focus();
}

// Enter для отправки
inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        send();
    }
});

sendBtn.addEventListener("click", send);

checkStatus();