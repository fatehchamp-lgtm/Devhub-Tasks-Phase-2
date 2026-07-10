import streamlit as st
import numpy as np
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set up the title and header of your web application
st.set_page_config(page_title="AI News Classifier", page_icon="📰", layout="centered")
st.title("📰 News Topic Classifier Using BERT")
st.write("An advanced AI automation task for **DevelopersHub Corporation Internship Phase 2**.")
st.markdown("---")

# AG News dataset mapping classes
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

# Load the saved fine-tuned model and tokenizer
MODEL_PATH = "./Task 1 - News Classifier/saved_bert_model"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"⚠️ Model directory '{MODEL_PATH}' not found. Please check your path!")
        return None, None
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model


tokenizer, model = load_model()

# User Input Interface
st.subheader("Enter a News Headline:")
user_input = st.text_area("Type or paste a news snippet here to classify:",
                          placeholder="e.g., NASA's rover discovers liquid water on Mars...")

if st.button("🚀 Classify Headline"):
    if user_input.strip() == "":
        st.warning("Please enter some text before clicking classify!")
    elif model is None:
        st.error("Cannot classify because the fine-tuned BERT model hasn't been saved yet.")
    else:
        with st.spinner("Analyzing headline with BERT..."):
            # Tokenize user input text
            inputs = tokenizer(user_input, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                prediction = torch.argmax(logits, dim=-1).item()
                probabilities = torch.nn.functional.softmax(logits, dim=-1).numpy()[0]

            # Display results beautifully
            st.success(f"**Predicted Category:** {CLASS_NAMES[prediction]}")

            # Show confidence breakdown
            st.write("### 📊 Confidence Breakdown:")
            for name, prob in zip(CLASS_NAMES, probabilities):
                st.progress(float(prob))
                st.write(f"{name}: **{prob * 100:.2f}%**")