# config.py

class Config:
    """
    Configuration class for the RAG pipeline.
    Settings for directories, models, and pipeline parameters.
    """
    
    # --- Directories ---
    DATA_DIRECTORY = './data'
    VECTOR_DB_PATH = 'faiss_index'
    BM25_INDEX_PATH = 'bm25_index.pkl'

    # --- Models ---
    EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
    
    # --- LLM (for RAG) ---
    LLM_BASE_URL = "http://localhost:1234/v1"
    LLM_MODEL_NAME = "phi-3-mini-4k-instruct"
    LLM_API_KEY = "not-needed-for-local" # Fetch from env or config in real scenarios

    # --- Ingestion Parameters ---
    MODEL_SAFE_CHUNK_SIZE = 1000
    MODEL_SAFE_CHUNK_OVERLAP = 200
    INGESTION_BATCH_SIZE = 100

    # --- Retriever Parameters ---
    FAISS_RETRIEVER_K = 2 # Number of results from FAISS
    BM25_RETRIEVER_K = 2  # Number of results from BM25
    ENSEMBLE_WEIGHTS = [0.5, 0.5] # Weights for [BM25, FAISS]