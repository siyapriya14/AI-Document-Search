from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM

import tempfile
import os

# Dummy LLM (academic demo – avoids Ollama / API dependency)
class SimpleLLM(LLM):
    @property
    def _llm_type(self):
        return "simple-llm"

    def _call(self, prompt, stop=None):
        return "This is a demo RAG response generated from the uploaded documents."

def get_answer_from_docs(query, uploaded_files):
    documents = []

    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        documents.extend(loader.load())

        os.remove(tmp_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(splits, embeddings)

    retriever = vectorstore.as_retriever()

    llm = SimpleLLM()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    result = qa_chain.run(query)
    return result
