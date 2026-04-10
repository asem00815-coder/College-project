let history = [];

const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send-btn");
const statusEl = document.getElementById("status");

async function checkStatus() {
    try {
        const data = await getStatus();
        if (data.total_chunks > 0) {
            statusEl.style.color = 'green'
            statusEl.textContent = "SUCCESS";
        } else {
            statusEl.style.color = 'red'
            statusEl.textContent = "ERROR";
        }
    } catch {
        statusEl.style.color = 'red'
        statusEl.textContent = "ERROR";
    }
}

function addMessage(text, role, sources = []) {
    const div = document.createElement("div");
    div.className = `message ${role}`;

    let sourcesHtml = "";
    if (sources.length > 0) {
        sourcesHtml = `<div class="sources">Source: ${sources.join(", ")}</div>`;
    }

    div.innerHTML = `
        <div class="bubble">${text}</div>
        ${sourcesHtml}`;

    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addLoading() {
    const div = document.createElement("div");
    div.className = "message bot loading";
    div.innerHTML = `<div class="bubble"><span></span><span></span><span></span></div>`;
    div.id = "loading";
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function removeLoading() {
    const el = document.getElementById("loading");
    if (el) el.remove();
}

async function send() {
    const text = inputEl.value.trim();
    if (!text) return;

    inputEl.value = "";
    sendBtn.disabled = true;

    addMessage(text, "user");
    history.push({role: "user", content: text});

    addLoading();
    inputEl.focus()

    try {
        const data = await sendMessage(text, history);
        removeLoading();
        addMessage(data.answer, "bot", data.sources);
        history.push({role: "assistant", content: data.answer});
    } catch (e) {
        removeLoading();
        addMessage("Ошибка соединения с сервером.", "bot");
    }

    sendBtn.disabled = false;
}

inputEl.addEventListener("input", () => {
    inputEl.style.height = "auto";
    inputEl.style.height = inputEl.scrollHeight + "px";
});

sendBtn.addEventListener("click", send);
checkStatus();