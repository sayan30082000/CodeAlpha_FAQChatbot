"""Text preprocessing with NLTK: lowercase -> tokenize -> remove punctuation & stopwords -> lemmatize."""
import re
import string
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Keep the NLTK data next to the project so it works the same on every machine.
NLTK_DIR = Path(__file__).parent / "nltk_data"
nltk.data.path.insert(0, str(NLTK_DIR))

_REQUIRED = {
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}


def ensure_nltk_data() -> None:
    """Download the NLTK resources on first run (only once)."""
    for package, resource in _REQUIRED.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            NLTK_DIR.mkdir(exist_ok=True)
            nltk.download(package, download_dir=str(NLTK_DIR), quiet=True)


ensure_nltk_data()

_lemmatizer = WordNetLemmatizer()
# Question words carry meaning in an FAQ ("how", "when", "where"), and so does "not".
_KEEP = {"how", "when", "where", "why", "what", "which", "not", "no", "can"}
_STOPWORDS = set(stopwords.words("english")) - _KEEP
_PUNCT = set(string.punctuation) | {"’", "‘", "“", "”", "—", "–", "..."}


def normalize(text: str) -> str:
    """Lowercase and strip punctuation / extra spaces (keeps every word)."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess(text: str) -> list[str]:
    """Return the cleaned, lemmatized tokens of `text`."""
    tokens = word_tokenize(text.lower())
    cleaned = []
    for tok in tokens:
        if tok in _PUNCT or tok in _STOPWORDS or not any(ch.isalnum() for ch in tok):
            continue
        tok = _lemmatizer.lemmatize(tok)           # noun form: "orders" -> "order"
        tok = _lemmatizer.lemmatize(tok, pos="v")  # verb form: "shipping" -> "ship"
        cleaned.append(tok)
    return cleaned


if __name__ == "__main__":
    sample = "How long does shipping take for my orders?"
    print(sample)
    print(preprocess(sample))
