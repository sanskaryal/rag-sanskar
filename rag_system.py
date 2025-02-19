# -*- coding: utf-8 -*-
"""
Updated RAG System with Latest LangChain/HuggingFace Syntax
"""

# Install required packages
# pip install langchain faiss-cpu sentence-transformers wikipedia huggingface_hub
# pip install -U langchain-community langchain-huggingface

from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint  # Updated import

def main():
    # 1. Data Preparation - Load documents from Wikipedia
    print("Loading documents...")
    loader = WikipediaLoader(query="Liam Payne", load_max_docs=1)
    documents = loader.load()
    
    # 2. Document Processing - Split into chunks
    print("Processing documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    # 3. Embedding Setup
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # 4. Vector Store
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # 5. Retriever Setup
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # 6. LLM Setup with updated configuration
    from langchain_huggingface import HuggingFaceEndpoint

    class DebugHuggingFaceEndpoint(HuggingFaceEndpoint):
        def _call(self, prompt, stop=None, **kwargs):
            print("\nDebug _call: Prompt being sent to HuggingFace LLM:")
            print(prompt)
            return super()._call(prompt, stop, **kwargs)
        
        def generate(self, prompt, stop=None, **kwargs):
            print("\nDebug generate: Prompt being sent to HuggingFace LLM:")
            print(prompt)
            return super().generate(prompt, stop, **kwargs)

    llm = DebugHuggingFaceEndpoint(
        repo_id="google/flan-t5-xxl",
        task="text-generation",  # Explicit task definition
        temperature=0.7,
        max_new_tokens=256  # Updated parameter name
    )
    # print("Initializing language model...")
    # llm = HuggingFaceEndpoint(
    #     repo_id="google/flan-t5-xxl",
    #     task="text-generation",  # Explicit task definition
    #     temperature=0.7,
    #     max_new_tokens=256  # Updated parameter name
    # )
    
    # 7. RAG Pipeline with invoke() pattern
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    
    # 8. Updated interactive query system
    print("\nRAG System Ready! Type 'exit' to quit.")
    while True:
        query = input("\nYour question: ")
        if query.lower() == 'exit':
            break
            
        # 9. Use invoke() instead of __call__
        result = qa_chain.invoke({"query": query})
            
        # 10. Display Results
        print(f"\nAnswer: {result['result']}")



if __name__ == "__main__":
    main()