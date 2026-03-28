from langchain_ollama import ChatOllama

def answer_query(query, retriever):
    # Initialize local LLM (Mistral via Ollama)
    llm = ChatOllama(
        model="mistral"
    )
    
    # Retrieve relevant documents
    docs = retriever.invoke(query)
    
    # Combine retrieved content
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create prompt
    prompt = f"""
You are an AI assistant. Answer the question ONLY using the provided context.

Context:
{context}

Question:
{query}

Answer:
"""
    
    # Get response from LLM
    response = llm.invoke(prompt)
    
    return response.content, docs