import streamlit as st
from transformers import pipeline

model_name = "google-t5/t5-small"


# Load the model only once; without this, Streamlit would reload it on every click
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model=model_name)


st.title("ISOM5160 Text Summarization")
st.write("Paste an English article and the app will write a short summary.")

article = st.text_area("Enter your article", height=250)

if st.button("Summarize"):
    if not article.strip():
        st.warning("Please enter an article first.")
    else:
        with st.spinner("Summarizing..."):
            summarizer = load_summarizer()
            result = summarizer(article, max_new_tokens=100, min_length=30)
            summary = result[0]["summary_text"]

        st.write(f"Model: {model_name}")
        st.write(f"Original length: {len(article.split())} words")
        st.write(f"Summary length: {len(summary.split())} words")
        st.subheader("Summary")
        st.write(summary)
