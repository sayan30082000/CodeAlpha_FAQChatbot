"""Web chat UI for the FAQ chatbot — CodeAlpha AI Internship, Task 2.

Run:  python app.py   then open http://127.0.0.1:5001
"""
from flask import Flask, jsonify, render_template, request

from chatbot import FAQBot

app = Flask(__name__)
bot = FAQBot()


@app.get("/")
def index():
    return render_template("index.html", store_name=bot.store_name, popular=bot.popular_questions())


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "")[:500]
    return jsonify(bot.ask(message).to_dict())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
