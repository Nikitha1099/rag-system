from langchain_community.document_loaders import PyPDFLoader
import os

def load_pdfs(data_path="data"):
    documents = []
    
    for file in os.listdir(data_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_path, file))
            docs = loader.load()
            documents.extend(docs)
    
    return documents