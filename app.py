import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from rag_pipeline import get_answer_from_docs

st.title("AI Document Search using RAG")

st.subheader("Upload a document (PDF or TXT)")

uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "txt"]
)

documents = None

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents = loader.load()

    elif uploaded_file.name.endswith(".txt"):
        loader = TextLoader(file_path)
        documents = loader.load()

    st.success("Document uploaded successfully!")

st.divider()

if documents is None:
    st.info("Please upload a document to ask questions.")
    st.stop()

query = st.text_input("Ask a question from the uploaded document")

if query:
    answer = get_answer_from_docs(documents, query)
    st.success(answer)
