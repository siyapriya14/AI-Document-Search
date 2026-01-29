import streamlit as st
from rag_pipeline import get_answer

st.title("📄 AI Document Search using RAG (Demo)")

query = st.text_input("Ask a question from the document:")

if query:
    answer = get_answer(query)
    st.success(answer)