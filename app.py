import streamlit as st
from rag_pipeline import get_answer_from_docs
import os

st.title("AI Document Search using RAG (Demo)")

# Upload file
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)

query = st.text_input("Ask a question from the document:")

if uploaded_file and query:
    # Save uploaded file
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    answer = get_answer_from_docs(query, file_path)
    st.success(answer)
