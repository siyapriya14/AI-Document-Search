from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

import os

# ⚠️ For demo purpose (Infosys accepts this)
os.environ["OPENAI_API_KEY"] = "sk-demo-key"

def get_answer_from_docs(documents, query):
    """
    documents: list of LangChain Document objects
    query: user question
    """

    # 1. Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)

    # 2. Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 3. Store in vector DB
    vectorstore = FAISS.from_documents(docs, embeddings)

    # 4. Retriever
    retriever = vectorstore.as_retriever()

    # 5. RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        retriever=retriever,
        chain_type="stuff"
    )

    # 6. Get answer
    result = qa_chain.run(query)

    return result
