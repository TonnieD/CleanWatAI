from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import re
import string
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

NLTK_PATH = os.path.join("app", "nltk_data")
os.makedirs(NLTK_PATH, exist_ok=True)

# Set both runtime path and environment variable
nltk.data.path.append(NLTK_PATH)
os.environ["NLTK_DATA"] = NLTK_PATH

# List of required resources
REQUIRED_NLTK_RESOURCES = [
    ("tokenizers/punkt", "punkt"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("tokenizers/punkt_tab", "punkt_tab"),
]

# Download if not already available
for path, name in REQUIRED_NLTK_RESOURCES:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(name, download_dir=NLTK_PATH)

# Then proceed to your pipeline setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#cleaning + lemmatization function
def clean_text(text):
    # Lowercase
    text = text.lower()
    
    # Handle negations (combine with next word if possible)
    text = re.sub(r"\b(no|not|never)\s+(\w+)", r"no_\2", text)

    # Remove punctuation except underscores (used in negation)
    text = re.sub(r"[^\w\s_]", "", text)

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Lemmatize and remove stopwords (except 'no_x' preserved words)
    cleaned = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if (token in stop_words and token.startswith('no_')) or (token not in stop_words and len(token) > 2)
    ]

    return " ".join(cleaned)

def clean_texts(texts):
    return [clean_text(text) for text in texts]