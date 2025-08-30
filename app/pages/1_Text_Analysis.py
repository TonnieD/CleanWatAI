import streamlit as st
import joblib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Go up to CleanWatAI/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
MODEL_PATH = PROJECT_ROOT / "app" / "models" / "nlp_pipeline.joblib"

from app.utils.text_cleaning import clean_text

# --- Load pipeline once and cache it ---
@st.cache_resource
def load_pipeline():
    if not MODEL_PATH.exists():
        st.error("⚠️ Trained pipeline not found. Please train and save the model first.")
        return None
    
    try:
        pipeline = joblib.load(MODEL_PATH)
        return pipeline
    except Exception as e:
        st.error(f"❌ Error loading pipeline: {e}")
        return None

pipeline = load_pipeline()

# --- Streamlit page ---
st.title("📝 Text Analysis")

st.write("Paste some text below and the trained NLP pipeline will predict severity.")

user_input = st.text_area("Enter text here:", height=200)

if st.button("Analyze Text"):
    if pipeline is None:
        st.error("Pipeline not available. Please retrain the model.")
    elif not user_input.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        # Clean & predict
        cleaned = clean_text(user_input)
        prediction = pipeline.predict([cleaned])[0]
        probabilities = pipeline.predict_proba([cleaned])[0]

        st.subheader("🔎 Prediction Result")
        st.write(f"**Predicted Severity:** {prediction}")

        st.subheader("📊 Probabilities")
        for label, prob in zip(pipeline.classes_, probabilities):
            st.write(f"- {label}: {prob:.4f}")
