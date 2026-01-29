from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import tempfile
import os

def get_answer_from_docs(query, uploaded_files):
    documents = []

    # Load PDFs
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            path = tmp.name

        loader = PyPDFLoader(path)
        documents.extend(loader.load())
        os.remove(path)

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Retrieve top docs
    docs = vectorstore.similarity_search(query, k=3)

    # Simple answer (demo-safe)
    answer = "Answer generated from uploaded documents:\n\n"
    for d in docs:
        answer += d.page_content[:500] + "\n\n"

    return answer
