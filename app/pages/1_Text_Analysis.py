import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import re
import sys
import os
from utils.text_cleaning import clean_texts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Define a single, persistent location for NLTK data
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


# Models and pipeline loading
@st.cache_resource
def load_nlp_model():
    base_dir = Path(__file__).resolve().parent.parent
    path = base_dir / "models" / "nlp_pipeline.pkl"
    return joblib.load(path)

nlp_pipeline = load_nlp_model() # NLP pipeline

st.title("🧠 NLP-Based Water Report Classification")
st.markdown("Use the form below to classify water safety based on textual observations.")

col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    user_text = st.text_area(label="Describe what you want to know", height=150)

    col1, col2, col3 = st.columns(3)
    with col1:
        color = st.selectbox("Water Color", options=["", "Clear", "Brown", "Green", "Other"])
    with col2:
        clarity = st.selectbox("Clarity", options=["", "Clear", "Murky"])
    with col3:
        odor = st.selectbox("Odor", options=["", "None", "Chemical", "Sewage", "Other"])

    col1, col2, col3 = st.columns(3)
    with col1:
        rain = st.selectbox("Recent Rain", options=["", "No recent rain", "Light rain", "Heavy rain"])
    with col2:
        activity = st.selectbox("Nearby Activity", options=["", "Residential", "Industrial", "Agricultural", "None"])
    with col3:
        infrastructure = st.selectbox("Infrastructure", options=["", "Good condition", "Needs repair", "Unknown"])
    context_parts = []

    if color:
        context_parts.append(f"The water appears {color.lower()} in color.")
    if clarity:
        context_parts.append(f"It is {clarity.lower()} in clarity.")
    if odor:
        context_parts.append(f"It has a {odor.lower()} odor.")

    if rain:
        context_parts.append(f"There was {rain.lower()}.")
    if activity:
        context_parts.append(f"The area nearby is {activity.lower()}.")
    if infrastructure:
        context_parts.append(f"The infrastructure is (in) {infrastructure.lower()}.")

    # Combine original input + context
    combined_description = user_text.strip() + " " + " ".join(context_parts)

    # Edited text area with combined description from the select boxes and user input
    edited_description = st.text_area("📝 Final Input to the Model (Editable)", value=combined_description, height=200)

    col1, col2, col3 = st.columns(3)
    with col3:
        if st.button("Submit", type="primary", use_container_width=True):
            # Only runs when Submit is clicked
            if not edited_description or edited_description.strip() == "":
                st.warning("Please describe your concern in the text area above.")
            else:
                prediction = nlp_pipeline.predict([edited_description])[0]
                probability = nlp_pipeline.predict_proba([edited_description])[0][prediction]

                # Map prediction to label
                label_map = {0: "Safe", 1: "Unsafe"}
                prediction_label = label_map[prediction]

                # Display result with appropriate style
                if prediction_label == "Safe":
                    st.success(f"✅ Water is predicted to be SAFE.\nConfidence: {probability:.2%}")
                else:
                    st.error(f"⚠️ Water is predicted to be UNSAFE.\nConfidence: {probability:.2%}")

with st.container(border=True):
        st.caption("© 2025 CleanWaterAI. Data sourced from WPDx and other public datasets.")