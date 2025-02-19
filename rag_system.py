import os
import faiss
import numpy as np
import wikipedia
from sentence_transformers import SentenceTransformer
from google import genai
import re

def chunk_text(text, chunk_size=1000, overlap=200):
    """Basic text chunking with overlap"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def main():
    # 1. Data Preparation
    print("Loading documents...")
    wikipedia.set_lang("en")
    page = wikipedia.page("Liam Payne", auto_suggest=False)
    text = page.content
    
    # 2. Document Processing
    print("Processing documents...")
    clean_text = re.sub('\s+', ' ', text).strip()
    chunks = chunk_text(clean_text, chunk_size=1000, overlap=200)
    
    # 3. Embedding Setup
    print("Creating embeddings...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True).cpu().numpy()
    
    # 4. Create FAISS Index
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_embeddings)
    
    # 5. Gemini Setup
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    print("\nRAG System Ready! Type 'exit' to quit.")
    while True:
        query = input("\nYour question: ")
        if query.lower() == 'exit':
            break
        
        # Retrieve relevant chunks
        query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
        indices = index.search(query_embedding, k=3)
        
        # Build context
        context = "\n\n".join([chunks[i] for i in indices[0]])
        
        # Generate response
        prompt = f"""Answer based on this context:
        {context}
        
        Question: {query}
        Answer:"""
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        print(f"\nAnswer: {response.text}")

if __name__ == "__main__":
    main()