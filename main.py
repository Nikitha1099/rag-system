from dotenv import load_dotenv
import os

load_dotenv()

from ingestion.load_data import load_pdfs
from ingestion.chunking import chunk_documents
from retrieval.vector_store import create_vector_store, get_retriever
from retrieval.qa import answer_query

def main():
    print("Loading documents...")
    docs = load_pdfs()
    
    print("Chunking...")
    chunks = chunk_documents(docs)
    
    print("Creating vector DB...")
    vectordb = create_vector_store(chunks)
    
    retriever = get_retriever(vectordb)
    
    while True:
        query = input("\nAsk a question (or 'exit'): ")
        
        if query == "exit":
            break
        
        answer, docs = answer_query(query, retriever)
        
        print("\nAnswer:\n", answer)
        print("\n--- Retrieved Chunks ---")
        for d in docs:
            print(d.page_content[:200], "\n")

if __name__ == "__main__":
    main()