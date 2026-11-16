"""FAQ chatbot engine — matches a user question to the most similar FAQ with cosine similarity.

How it works
1. Every FAQ question (plus a few alternative phrasings) is preprocessed with NLTK.
2. Two TF-IDF vectorizers turn the text into vectors:
   - word level (unigrams + bigrams of the lemmatized tokens) captures meaning,
   - character level (3-5 char n-grams) makes matching robust to typos ("shiping").
3. A user question is vectorized the same way and compared to every phrasing with
   cosine similarity. The FAQ with the highest score wins.
4. If the best score is below a threshold the bot says it is not sure and suggests
   the closest FAQs instead of guessing.

Run in the terminal:  python chatbot.py
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp_utils import normalize, preprocess

DATA_FILE = Path(__file__).parent / "data" / "faqs.json"

WORD_WEIGHT = 0.6        # share of the word-level similarity in the final score
CHAR_WEIGHT = 0.4        # share of the character-level similarity
ANSWER_THRESHOLD = 0.35  # below this the bot asks for clarification
SMALLTALK_THRESHOLD = 0.8


@dataclass
class BotReply:
    answer: str
    matched_question: str | None = None
    category: str | None = None
    score: float = 0.0
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "matched_question": self.matched_question,
            "category": self.category,
            "score": round(self.score, 3),
            "suggestions": self.suggestions,
        }


def _join_tokens(text: str) -> str:
    return " ".join(preprocess(text))


class FAQBot:
    def __init__(self, data_file: Path = DATA_FILE):
        data = json.loads(Path(data_file).read_text(encoding="utf-8"))
        self.store_name = data.get("store_name", "our store")
        self.faqs = data["faqs"]
        self.smalltalk = data.get("smalltalk", [])

        # One row per phrasing; remember which FAQ each row belongs to.
        self._phrasings, self._owner = [], []
        for idx, faq in enumerate(self.faqs):
            for text in [faq["question"], *faq.get("alternatives", [])]:
                self._phrasings.append(text)
                self._owner.append(idx)
        self._owner = np.array(self._owner)

        self._word_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self._word_matrix = self._word_vec.fit_transform(_join_tokens(t) for t in self._phrasings)

        self._char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self._char_matrix = self._char_vec.fit_transform(normalize(t) for t in self._phrasings)

        # Small talk ("hi", "thanks", "who are you") is matched on the raw normalized text,
        # because those sentences are almost entirely stopwords.
        self._st_patterns, self._st_owner = [], []
        for idx, item in enumerate(self.smalltalk):
            for pattern in item["patterns"]:
                self._st_patterns.append(normalize(pattern))
                self._st_owner.append(idx)
        if self._st_patterns:
            self._st_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
            self._st_matrix = self._st_vec.fit_transform(self._st_patterns)

    # ------------------------------------------------------------------ matching
    def _faq_scores(self, question: str) -> np.ndarray:
        """Best similarity score of `question` against each FAQ (max over its phrasings)."""
        char_sim = cosine_similarity(self._char_vec.transform([normalize(question)]), self._char_matrix)[0]
        tokens = _join_tokens(question)
        if tokens:
            word_sim = cosine_similarity(self._word_vec.transform([tokens]), self._word_matrix)[0]
            sim = WORD_WEIGHT * word_sim + CHAR_WEIGHT * char_sim
        else:
            sim = char_sim * CHAR_WEIGHT  # nothing meaningful left after preprocessing

        scores = np.zeros(len(self.faqs))
        np.maximum.at(scores, self._owner, sim)
        return scores

    def _match_smalltalk(self, question: str) -> BotReply | None:
        text = normalize(question)
        if not self._st_patterns or not text or len(text.split()) > 6:
            return None
        if text in self._st_patterns:
            item = self.smalltalk[self._st_owner[self._st_patterns.index(text)]]
            return BotReply(answer=item["answer"], category="Small talk", score=1.0)
        sim = cosine_similarity(self._st_vec.transform([text]), self._st_matrix)[0]
        best = int(sim.argmax())
        if sim[best] >= SMALLTALK_THRESHOLD:
            item = self.smalltalk[self._st_owner[best]]
            return BotReply(answer=item["answer"], category="Small talk", score=float(sim[best]))
        return None

    def ask(self, question: str) -> BotReply:
        question = (question or "").strip()
        if not question:
            return BotReply(answer="Please type a question. 🙂")

        smalltalk = self._match_smalltalk(question)
        if smalltalk:
            return smalltalk

        scores = self._faq_scores(question)
        ranked = np.argsort(scores)[::-1]
        best = int(ranked[0])
        best_score = float(scores[best])

        if best_score >= ANSWER_THRESHOLD:
            faq = self.faqs[best]
            related = [self.faqs[i]["question"] for i in ranked[1:3] if scores[i] >= ANSWER_THRESHOLD * 0.8]
            return BotReply(
                answer=faq["answer"],
                matched_question=faq["question"],
                category=faq["category"],
                score=best_score,
                suggestions=related,
            )

        suggestions = [self.faqs[i]["question"] for i in ranked[:3] if scores[i] > 0.1]
        answer = "Sorry, I'm not sure I understood that."
        answer += " Did you mean one of these?" if suggestions else " Could you rephrase your question?"
        return BotReply(answer=answer, score=best_score, suggestions=suggestions)

    def popular_questions(self, n: int = 6) -> list[str]:
        picks = ["track_order", "shipping_time", "return_policy", "payment_methods", "refund_time", "contact_support"]
        by_id = {faq["id"]: faq["question"] for faq in self.faqs}
        return [by_id[p] for p in picks if p in by_id][:n]


def main() -> None:
    bot = FAQBot()
    print(f"{bot.store_name} FAQ Chatbot — type your question (or 'quit' to exit)\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        reply = bot.ask(question)
        print(f"Bot: {reply.answer}")
        if reply.matched_question:
            print(f"     (matched: \"{reply.matched_question}\" — score {reply.score:.2f})")
        for s in reply.suggestions if not reply.matched_question else []:
            print(f"     • {s}")
        print()
    print("Bot: Goodbye!")


if __name__ == "__main__":
    main()
