const messages = document.getElementById("messages");
const form = document.getElementById("form");
const input = document.getElementById("message");

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function addMessage(role, text, meta) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.appendChild(bubble);
  if (meta) {
    const info = document.createElement("div");
    info.className = "meta";
    info.textContent = meta;
    wrap.appendChild(info);
  }
  messages.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function addChips(questions, label) {
  if (!questions || !questions.length) return;
  if (label) {
    const note = document.createElement("div");
    note.className = "meta";
    note.textContent = label;
    messages.appendChild(note);
  }
  const chips = document.createElement("div");
  chips.className = "chips";
  questions.forEach((q) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = q;
    chips.appendChild(chip);
  });
  messages.appendChild(chips);
  scrollToBottom();
}

function showTyping() {
  const wrap = document.createElement("div");
  wrap.className = "msg bot typing";
  wrap.innerHTML = '<div class="bubble"><span></span><span></span><span></span></div>';
  messages.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

// Two backends with the same answer format:
//  - the Python chatbot behind Flask (/api/chat) when you run app.py
//  - the in-browser engine (engine.js + model.json) on the static live site
async function askBot(text) {
  if (window.FAQEngine) return window.FAQEngine.askAsync(text);
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: text }),
  });
  return res.json();
}

async function send(text) {
  text = text.trim();
  if (!text) return;
  addMessage("user", text);
  input.value = "";
  const typing = showTyping();

  try {
    const [data] = await Promise.all([
      askBot(text),
      new Promise((r) => setTimeout(r, 450)), // short pause so the typing dots are visible
    ]);
    typing.remove();

    const meta = data.matched_question
      ? `Matched FAQ: "${data.matched_question}" · ${data.category} · ${Math.round(data.score * 100)}% match`
      : null;
    addMessage("bot", data.answer, meta);

    if (data.matched_question) addChips(data.suggestions, "Related questions:");
    else addChips(data.suggestions);
  } catch {
    typing.remove();
    addMessage("bot", "Sorry, I couldn't reach the server. Please try again.");
  }
  input.focus();
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  send(input.value);
});

// Clicking any suggestion chip asks that question.
messages.addEventListener("click", (e) => {
  if (e.target.classList.contains("chip")) send(e.target.textContent);
});
