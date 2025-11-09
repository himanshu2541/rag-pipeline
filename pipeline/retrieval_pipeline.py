import sys
import time
import os

from components.embedding_model import get_embedding_model
from config import VECTOR_DB_PATH, EMBEDDING_MODEL_NAME

# Import the vector store loader
from langchain_community.vectorstores import FAISS

def query_vector_store(query_text):
    """
    Loads the vector store and queries it with the given text.
    """
    print(f"\n--- Starting Retrieval ---")
    
    # 1. Check if the vector store exists
    if not os.path.exists(VECTOR_DB_PATH):
        print(f"Error: Vector store not found at {VECTOR_DB_PATH}.")
        print("Please run the ingestion pipeline first with: python main.py --ingest")
        return

    # 2. Load the Embedding Model
    #    (Must be the *same* model used for ingestion)
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embeddings = get_embedding_model(EMBEDDING_MODEL_NAME)
    print(f"Embedding model loaded")

    print(f"Loading vector store from {VECTOR_DB_PATH}...")
    vector_store = FAISS.load_local(
        VECTOR_DB_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    print("Vector store loaded successfully.")

    print(f"\nQuerying for: '{query_text}'")
    start_time = time.time()
    results = vector_store.similarity_search(query_text, k=3)
    end_time = time.time()
    print(f"Query completed in {end_time - start_time:.2f}s")

    if not results:
        print("No results found.")
        return

    print(f"\nFound {len(results)} relevant chunks:")
    for i, doc in enumerate(results):
        print("---------------------------------")
        print(f"Result {i+1} (from: {doc.metadata.get('source', 'Unknown')}):")
        print(doc.page_content)
        print("---------------------------------")

def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        query_vector_store(query)
    else:
        print("No query provided. Running with a default test query.")
        print("Usage: python retrieval_pipeline.py \"Your query here\"")
        query_vector_store("What is RAG?")

if __name__ == "__main__":
    main()
