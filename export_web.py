"""Build the browser version of the chatbot in web/ (used for the live demo on Netlify).

Netlify can only host static files, so this script exports everything the Python
chatbot learned into web/model.json, and web/engine.js repeats the same matching
in JavaScript:
  - the NLTK stopword list and a word -> lemma table (built with NLTK's lemmatizer)
  - both TF-IDF vectorizers (vocabulary + IDF weights) and the FAQ vectors
  - the small-talk patterns, thresholds and weights
It also renders the chat page and copies the CSS/JS from static/.

    python export_web.py
"""
import json
import re
import shutil
from pathlib import Path

from nltk.corpus import wordnet

import chatbot
from app import app, bot
from nlp_utils import _STOPWORDS, normalize, preprocess

ROOT = Path(__file__).parent
WEB = ROOT / "web"


def sparse_rows(matrix) -> list:
    """CSR matrix -> [[indices], [values]] per row (values rounded to keep the file small)."""
    m = matrix.tocsr()
    rows = []
    for r in range(m.shape[0]):
        start, end = m.indptr[r], m.indptr[r + 1]
        rows.append([m.indices[start:end].tolist(), [round(float(v), 6) for v in m.data[start:end]]])
    return rows


def vectorizer_spec(vec) -> dict:
    return {
        "vocab": {term: int(idx) for term, idx in vec.vocabulary_.items()},
        "idf": [round(float(v), 6) for v in vec.idf_],
        "ngram_range": list(vec.ngram_range),
        "sublinear_tf": bool(vec.sublinear_tf),
    }


def inflections(lemma: str) -> set[str]:
    """Common English inflected forms of a word (plural, past, -ing ...)."""
    forms = {lemma, lemma + "s", lemma + "es", lemma + "ed", lemma + "d", lemma + "ing", lemma + "er"}
    if lemma.endswith("e"):
        forms |= {lemma[:-1] + "ing", lemma[:-1] + "ed"}
    if lemma.endswith("y") and len(lemma) > 2:
        forms |= {lemma[:-1] + "ies", lemma[:-1] + "ied"}
    if re.fullmatch(r".*[^aeiou][aeiou][bdgklmnprt]", lemma):
        forms |= {lemma + lemma[-1] + "ing", lemma + lemma[-1] + "ed"}
    return forms


def build_lemma_table(vocab_words: set[str], corpus_words: set[str]) -> dict:
    """surface word -> lemma, computed with the same NLTK preprocessing the Python bot uses."""
    candidates = set(corpus_words)
    for lemma in vocab_words:
        candidates |= inflections(lemma)
    # irregular forms from WordNet's exception lists (e.g. "bought" -> "buy")
    for pos in ("n", "v"):
        for surface, bases in wordnet._exception_map[pos].items():
            if any(b in vocab_words for b in bases):
                candidates.add(surface)

    table = {}
    for word in sorted(candidates):
        if "_" in word or " " in word:
            continue
        tokens = preprocess(word)
        if len(tokens) == 1 and tokens[0] != word:
            table[word] = tokens[0]
    return table


def main():
    WEB.mkdir(exist_ok=True)

    word_vec, char_vec = bot._word_vec, bot._char_vec
    vocab_words = {t for t in word_vec.vocabulary_ if " " not in t}
    corpus_text = " ".join(
        [faq["question"] + " " + " ".join(faq.get("alternatives", [])) for faq in bot.faqs]
        + [p for item in bot.smalltalk for p in item["patterns"]]
    )
    corpus_words = set(re.findall(r"[a-z]+", corpus_text.lower()))

    model = {
        "store_name": bot.store_name,
        "popular": bot.popular_questions(),
        "faqs": [{k: faq[k] for k in ("id", "category", "question", "answer")} for faq in bot.faqs],
        "smalltalk": [{"id": s["id"], "answer": s["answer"]} for s in bot.smalltalk],
        "settings": {
            "word_weight": chatbot.WORD_WEIGHT,
            "char_weight": chatbot.CHAR_WEIGHT,
            "answer_threshold": chatbot.ANSWER_THRESHOLD,
            "smalltalk_threshold": chatbot.SMALLTALK_THRESHOLD,
        },
        "stopwords": sorted(_STOPWORDS),
        "lemmas": build_lemma_table(vocab_words, corpus_words),
        "word": {**vectorizer_spec(word_vec), "rows": sparse_rows(bot._word_matrix)},
        "char": {**vectorizer_spec(char_vec), "rows": sparse_rows(bot._char_matrix)},
        "owner": bot._owner.tolist(),
        "smalltalk_patterns": bot._st_patterns,
        "smalltalk_owner": bot._st_owner,
        "smalltalk_char": {**vectorizer_spec(bot._st_vec), "rows": sparse_rows(bot._st_matrix)},
    }
    (WEB / "model.json").write_text(json.dumps(model, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    # The chat page: same template and assets as the Flask app, plus the in-browser engine.
    with app.test_request_context():
        html = app.jinja_env.get_template("index.html").render(
            store_name=bot.store_name, popular=bot.popular_questions(), url_for=lambda _, filename: filename
        )
    html = html.replace('<script src="app.js"></script>', '<script src="engine.js"></script>\n  <script src="app.js"></script>')
    (WEB / "index.html").write_text(html, encoding="utf-8")
    for name in ("style.css", "app.js"):
        shutil.copy(ROOT / "static" / name, WEB / name)

    size_kb = (WEB / "model.json").stat().st_size / 1024
    print(f"web/model.json: {size_kb:.0f} KB | {len(model['lemmas'])} lemma entries | "
          f"{len(model['word']['vocab'])} word terms | {len(model['char']['vocab'])} char n-grams")
    print("web/index.html, style.css, app.js updated. engine.js is hand-written.")


if __name__ == "__main__":
    main()
