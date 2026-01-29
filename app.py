import streamlit as st
from rag_pipeline import get_answer_from_docs

st.title("AI Document Search using RAG (Demo)")

query = st.text_input("Ask a question from the document:")

if query:
    answer = get_answer_from_docs(query)
    st.success(answer)
