import streamlit as st
import re
import string
import joblib

# Load saved vectorizer and model
vectorizer = joblib.load('tfidf_vectorizer.joblib')
model = joblib.load('calibrated_lr_model.joblib')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.title("Fake News Detection System")
st.write("Enter a news article below to check if it is Real or Fake.")

user_input = st.text_area("News Article")

if st.button("Check News"):
    if not user_input.strip():
        st.error("Please enter some news text to check.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]

        if prediction == 0:
            st.success("🟢 This news is predicted to be REAL.")
        else:
            st.error("🔴 This news is predicted to be FAKE.")

        st.write(f"Probability Real: {proba[0]:.4f}")
        st.write(f"Probability Fake: {proba[1]:.4f}")
