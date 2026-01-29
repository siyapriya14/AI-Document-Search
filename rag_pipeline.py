# rag_pipeline.py
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_answer(query: str) -> str:
    # Load document
    loader = TextLoader("data/test.txt", encoding="utf-8")
    documents = loader.load()

    # Split document
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # Simple retrieval (keyword match)
    query_lower = query.lower()
    for chunk in chunks:
        if query_lower in chunk.page_content.lower():
            return chunk.page_content

    return "Answer not found in the document. This is a demo RAG response."