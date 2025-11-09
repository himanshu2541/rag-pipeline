import sys
import time
import os

from components.embedding_model import get_embedding_model
from config import VECTOR_DB_PATH, EMBEDDING_MODEL_NAME

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


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

    # 2a. Initialize FAISS (Semantic Retriver)
    #    (Must be the *same* model used for ingestion)
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embeddings = get_embedding_model(EMBEDDING_MODEL_NAME)
    print(f"Embedding model loaded")

    print(f"Loading vector store from {VECTOR_DB_PATH}...")
    vector_store = FAISS.load_local(
        VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True
    )
    faiss_retriver = vector_store.as_retriever(search_kwargs={"k": 3})
    print("FAISS retriver loaded.")

    # 2b. Initialize BM25 (keyword retriever)
    print("Extracting chunks from FAISS docstore for BM25...")

    all_chunks = []
    if not vector_store.index_to_docstore_id:
        print("Error: FAISS docstore appears to be empty.")
        return

    for doc_id in vector_store.index_to_docstore_id.values():
        all_chunks.append(vector_store.docstore.search(doc_id))

    if not all_chunks:
        print("Error: No chunks found in FAISS docstore to initialize BM25.")
        return

    print(f"Initializing BM25 retriever with {len(all_chunks)} chunks from FAISS...")
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = 3
    print("BM25 retriever initialized.")

    # 3. Initialize EnsembleRetriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriver], weights=[0.5, 0.5]
    )
    print("Ensemble retriever created.")

    # 4. Query
    print(f"\nQuerying for: '{query_text}'")
    start_time = time.time()
    results = ensemble_retriever.invoke(query_text)
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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        query_vector_store(query)
    else:
        print("No query provided. Running with a default test query.")
        print('Usage: python retrieval_pipeline.py "Your query here"')
        query_vector_store("What is RAG?")
