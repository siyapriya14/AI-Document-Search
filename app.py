import streamlit as st
from rag_pipeline import get_answer_from_docs

st.set_page_config(page_title="AI Document Search using RAG", layout="centered")

st.title("AI Document Search using RAG (Demo)")

uploaded_files = st.file_uploader(
    "Upload PDF document(s)",
    type=["pdf"],
    accept_multiple_files=True
)

query = st.text_input("Ask a question from the document:")

if st.button("Get Answer"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF document.")
    elif not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing documents and generating answer..."):
            try:
                answer = get_answer_from_docs(query, uploaded_files)
                st.success("Answer:")
                st.write(answer)
            except Exception as e:
                st.error("Something went wrong while processing the documents.")
                st.exception(e)
