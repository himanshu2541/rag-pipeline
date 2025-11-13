import os
import pickle
import time
from config import Config

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

class RetrieverProvider:
    """
    Responsible for building and providing the ensemble retriever.
    If the BM25 index is not pickled, it will be created from the FAISS docstore.
    """
    def __init__(self, config: Config, embeddings):
        self.config = config
        self.embeddings = embeddings
        
        if not os.path.exists(self.config.VECTOR_DB_PATH):
            raise FileNotFoundError(
                f"Vector store not found at {self.config.VECTOR_DB_PATH}. "
                "Please run the --ingest command first."
            )

    def _load_faiss_store(self):
        """
        Internal helper to load the FAISS vector store.
        """
        if not os.path.exists(self.config.VECTOR_DB_PATH):
            raise FileNotFoundError(f"Vector store not found at {self.config.VECTOR_DB_PATH}.")
        
        return FAISS.load_local(
            self.config.VECTOR_DB_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
    

    def _build_and_save_bm25(self, vector_store):
        """
        Build the BM25 index from the FAISS docstore and save it to a pickle file (one-time operation, Memory-intensive).
        """
        print("\nWARNING: BM25 index not found...")
        print(f"Building new BM25 index from FAISS docstore...")
        print("This is a one-time operation and may take a lot of memory.")
        start_time = time.perf_counter()

        # 1. Extract all chunks from FAISS
        all_chunks = []
        if not vector_store.index_to_docstore_id:
            raise ValueError("FAISS docstore appears to be empty. Cannot init BM25.")

        print(f"Extracting {len(vector_store.index_to_docstore_id)} chunks...")
        for doc_id in vector_store.index_to_docstore_id.values():
            all_chunks.append(vector_store.docstore.search(doc_id))
        
        if not all_chunks:
            raise ValueError("No chunks found in FAISS docstore to initialize BM25.")

        # 2. Build the retriever
        print(f"Initializing BM25 retriever with {len(all_chunks)} chunks...")
        bm25_retriever = BM25Retriever.from_documents(all_chunks)

        # 3. Save to pickle
        print(f"Saving BM25 index to {self.config.BM25_INDEX_PATH}...")
        with open(self.config.BM25_INDEX_PATH, "wb") as f:
            pickle.dump(bm25_retriever, f)
        
        end_time = time.perf_counter()
        print(f"BM25 index built and saved in {end_time - start_time:.2f}s.")
        
        return bm25_retriever
    
    def get_vector_store(self):
        """
        Public method to get the FAISS vector store.
        """
        return self._load_faiss_store()
    
    def get_retriever(self):
        """
        Loads and returns the ensemble retriever.
        """
        print("Initializing retriever...")

        # 1. Load FAISS Retriever
        print("Loading FAISS vector store...")
        vector_store = self._load_faiss_store()
        faiss_retriever = vector_store.as_retriever(
            search_kwargs={"k": self.config.FAISS_RETRIEVER_K}
        )

        # 2. Load or Build BM25 Retriever
        if os.path.exists(self.config.BM25_INDEX_PATH):
            print(f"Loading BM25 index from {self.config.BM25_INDEX_PATH}...")
            with open(self.config.BM25_INDEX_PATH, "rb") as f:
                bm25_retriever = pickle.load(f)
        else:
            # Pass the loaded vector_store to the builder
            bm25_retriever = self._build_and_save_bm25(vector_store)
        
        bm25_retriever.k = self.config.BM25_RETRIEVER_K

        # 3. Initialize EnsembleRetriever
        print("Creating ensemble retriever...")
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=self.config.ENSEMBLE_WEIGHTS,
        )
        
        print("Retriever initialized.")
        return ensemble_retriever