import re
import os
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# === Force NLTK data into project/app/nltk_data ===
PROJECT_ROOT = Path(__file__).resolve().parents[1].parent  # project/
NLTK_PATH = PROJECT_ROOT / "app" / "nltk_data"
NLTK_PATH.mkdir(parents=True, exist_ok=True)

# Tell both env + nltk to use only this path
os.environ["NLTK_DATA"] = str(NLTK_PATH)
nltk.data.path = [str(NLTK_PATH)]

# === Ensure required packages exist ===
REQUIRED = {
    "punkt": "tokenizers/punkt",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}

for name, path in REQUIRED.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(name, download_dir=str(NLTK_PATH))

# === NLP tools ===
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# === Cleaning functions ===
def clean_text(text: str) -> str:
    """
    Lowercases, handles negations, removes punctuation, tokenizes,
    lemmatizes, and removes stopwords.
    """
    text = text.lower()

    # Handle negations
    text = re.sub(r"\b(no|not|never)\s+(\w+)", r"no_\2", text)

    # Remove punctuation (keep underscores for negations)
    text = re.sub(r"[^\w\s_]", "", text)

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Lemmatize & filter
    cleaned = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if (
            token.startswith("no_")  # keep negations
            or (token not in stop_words and len(token) > 2)
        )
    ]
    return " ".join(cleaned)

def clean_texts(texts):
    return [clean_text(text) for text in texts]
