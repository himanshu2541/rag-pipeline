import os
import time
from config import Config
from typing import List, Callable

from components.document_loader import load_documents_lazy
from components.text_splitter import split_documents
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document


class Ingestor:
    """
    Responsible for the document ingestion pipeline.
    This process loads, splits, and create a persistent FAISS vector store.
    """

    def __init__(self, config: Config, embeddings):
        self.config = config
        self.embeddings = embeddings

    def _process_batch_for_store(
        self,
        doc_batch: List[Document],
        text_splitter_func: Callable[[List[Document]], List[Document]],
        vector_store: FAISS | None,
    ):
        """
            Internal helper to split, embed, and add a single batch to the FAISS store.
        """
        try:
            # 1. Split documents in the batch
            chunks = text_splitter_func(doc_batch)
            if not chunks:
                print("Warning: No chunks created for this batch.")
                return vector_store

            # 2. Add to vector store
            if vector_store is None:
                print("Initializing vector store with first batch...")
                vector_store = FAISS.from_documents(chunks, self.embeddings)
            else:
                vector_store.add_documents(chunks)

        except Exception as e:
            print(f"Error processing batch: {e}")

        return vector_store

    def run(self):
        """
        Runs the full document ingestion pipeline.
        """
        print("\nStarting Document Ingestion Pipeline...")
        start_time = time.perf_counter()

        # 1. Create data directory if it doesn't exist
        if not os.path.exists(self.config.DATA_DIRECTORY):
            print(f"Creating data directory: {self.config.DATA_DIRECTORY}")
            os.makedirs(self.config.DATA_DIRECTORY)

        # 2. Set up document loader
        print("Setting up lazy document loader and splitter...")
        document_iterator = load_documents_lazy(self.config.DATA_DIRECTORY)
        splitter_with_args = lambda docs: split_documents(
            docs,
            chunk_size=self.config.MODEL_SAFE_CHUNK_SIZE,
            chunk_overlap=self.config.MODEL_SAFE_CHUNK_OVERLAP,
        )

        # 3. Process documents in batches
        print(f"Starting batch ingestion (size: {self.config.INGESTION_BATCH_SIZE})...")
        vector_store = None
        doc_batch = []
        total_docs_processed = 0

        for doc in document_iterator:
            doc_batch.append(doc)
            if len(doc_batch) >= self.config.INGESTION_BATCH_SIZE:
                vector_store = self._process_batch_for_store(
                    doc_batch, splitter_with_args, vector_store
                )
                total_docs_processed += len(doc_batch)
                print(f"Processed {total_docs_processed} documents...")
                doc_batch = []
        
        # 4. Process any remaining documents in the last batch
        if doc_batch:
            vector_store = self._process_batch_for_store(
                doc_batch, splitter_with_args, vector_store
            )
            total_docs_processed += len(doc_batch)

        # 5. Save the final vector store
        if vector_store:
            vector_store.save_local(self.config.VECTOR_DB_PATH)
            print(f"\nFAISS index saved successfully to {self.config.VECTOR_DB_PATH}")
            print(f"Total documents processed: {total_docs_processed}")
            print("BM25 index will be created on first run.")
        else:
            print("\nPipeline FAILED: No documents were processed.")

        end_time = time.perf_counter()
        print(f"Total time taken: {end_time - start_time:.2f} seconds.")