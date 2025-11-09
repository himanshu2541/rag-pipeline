from components.embedding_model import get_embedding_model
from config import VECTOR_DB_PATH, EMBEDDING_MODEL_NAME

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


def get_retriever():
    """
    Helper function to create and return the ensemble retriever efficiently.
    """
    # 1. Initialize FAISS Retriever
    print("Loading FAISS vector store...")
    embeddings = get_embedding_model(EMBEDDING_MODEL_NAME)
    vector_store = FAISS.load_local(
        VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True
    )
    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # 2. Initialize BM25 Retriever
    print("Extracting chunks from FAISS docstore for BM25...")

    all_chunks = []
    if not vector_store.index_to_docstore_id:
        raise ValueError("FAISS docstore appears to be empty. Cannot init BM25.")

    for doc_id in vector_store.index_to_docstore_id.values():
        all_chunks.append(vector_store.docstore.search(doc_id))

    print(f"Initializing BM25 retriever with {len(all_chunks)} chunks from FAISS...")
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = 2

    # 3. Initialize EnsembleRetriever
    print("Creating ensemble retriever...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
    )
    return ensemble_retriever
