from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

def get_answer_from_docs(query):
    # Load documents
    loader = PyPDFLoader("data/sample.pdf")
    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Vector DB
    db = FAISS.from_documents(docs, embeddings)

    # Retrieve
    retriever = db.as_retriever()
    relevant_docs = retriever.get_relevant_documents(query)

    # Simple LLM response
    context = "\n".join([doc.page_content for doc in relevant_docs])

    llm = Ollama(model="llama2")
    prompt = f"Answer the question using the context below:\n\n{context}\n\nQuestion: {query}"

    return llm.invoke(prompt)
