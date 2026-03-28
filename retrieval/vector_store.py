from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_vector_store(chunks):
    embedding = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="db"
    )
    
    return vectordb


def get_retriever(vectordb):
    return vectordb.as_retriever(search_kwargs={"k": 3})