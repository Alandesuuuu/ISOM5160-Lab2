import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="ISOM5160 Sentiment App")


# Load the model only once; without this, Streamlit would reload it on every click
@st.cache_resource
def load_classifier():
    return pipeline("text-classification",
                    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")


st.title("ISOM5160 Sentiment Analysis")
st.write("Type an English sentence and the app will tell you whether it is positive or negative.")

text = st.text_area("Enter your text", placeholder="e.g. I love this course!")

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            classifier = load_classifier()
            result = classifier(text)[0]

        label = result["label"]
        score = float(result["score"])

        if label == "POSITIVE":
            st.success(f"Result: {label}")
        else:
            st.error(f"Result: {label}")

        st.write(f"Confidence: {score:.2%}")
        st.progress(score)
