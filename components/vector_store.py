from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from typing import List, Iterator, Callable


# Keeping this for now maybe useful
def create_vector_store(chunks, embedding_model, db_path="faiss_index"):
    """
    Creates and saves a vector store from document chunks.

    Args:
      chunks (list): The list of document chunks.
      embedding_model (object): The embedding model to use.
      db_path (str): The directory path to save the FAISS index to.

    Returns:
      FAISS: The created vector store object.
    """

    if not chunks:
        print("No chunks provided to create vector store.")
        return None

    print(f"Creating vector store with {len(chunks)} chunks....")

    try:
        # Create the vector store from the document chunks
        vector_store = FAISS.from_documents(chunks, embedding_model)

        # Save the vector store locally
        vector_store.save_local(db_path)

        print(f"Vector store created and saved successfully to {db_path}")
        return vector_store

    except Exception as e:
        print(f"Error creating or saving vector store: {e}")
        return None


def create_vector_store_from_iterator(
    document_iterator: Iterator[Document],
    text_splitter_func: Callable[[List[Document]], List[Document]],
    embedding_model,
    db_path="faiss_index",
    batch_size=100,
):
    """
    Creates and saves a vector store from a document iterator.

    Args:
        document_iterator (Iterator[Document]): Iterator yielding Document objects.
        text_splitter_func (Callable[[List[Document]], List[Document]]): Function to split or transform documents into chunks.
        embedding_model: Embedding model instance used to create embeddings for documents/chunks.
        db_path (str): Directory path to save the FAISS index to. Defaults to "faiss_index".
        batch_size (int): Number of documents/chunks to process per batch. Defaults to 100.

    Returns:
        FAISS: The created vector store object, or None on failure.
    """
    print(f"Starting batch ingestion to vector store (batch size: {batch_size})...")
    vector_store = None
    doc_batch = []
    total_docs_processed = 0

    for doc in document_iterator:
        doc_batch.append(doc)
        if len(doc_batch) >= batch_size:
            vector_store = _process_batch_for_store(
                doc_batch, text_splitter_func, embedding_model, vector_store
            )
            total_docs_processed += len(doc_batch)
            print(f"Processed {total_docs_processed} documents...")
            doc_batch = []

    # process any remaining documents in the last batch
    if doc_batch:
        vector_store = _process_batch_for_store(
            doc_batch, text_splitter_func, embedding_model, vector_store
        )
        total_docs_processed += len(doc_batch)

    # Save the final vector store
    if vector_store:
        vector_store.save_local(db_path)
        print(f"Vector store saved successfully to {db_path}")
        print(f"Total documents processed: {total_docs_processed}")
        return vector_store
    else:
        print("No documents were processed to create the vector store.")
        return None


def _process_batch_for_store(
    doc_batch: List[Document],
    text_splitter_func: Callable[[List[Document]], List[Document]],
    embedding_model,
    vector_store: FAISS | None,
):
    """
    Internal helper to split, embed, and add a single batch to the FAISS store.

    Args:
        doc_batch (List[Document]): Batch of Document objects to split and index.
        text_splitter_func (Callable[[List[Document]], List[Document]]): Function to split or transform documents into chunks.
        embedding_model: Embedding model instance used by FAISS.from_documents.
        vector_store (FAISS | None): Existing FAISS store to append to, or None to create a new one.

    Returns:
        FAISS: The created vector store object, or None on failure.
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
            vector_store = FAISS.from_documents(chunks, embedding_model)
        else:
            vector_store.add_documents(chunks)

    except Exception as e:
        print(f"Error processing batch: {e}")

    return vector_store


if __name__ == "__main__":
    # simple test
    from components.embedding_model import get_embedding_model
    from langchain_community.docstore.document import Document

    test_chunks = [
        Document(
            page_content="RAG stands for Retrieval-Augmented Generation.",
            metadata={"source": "doc1"},
        ),
        Document(
            page_content="A vector store indexes document embeddings.",
            metadata={"source": "doc2"},
        ),
    ]
    test_embeddings = get_embedding_model()

    print("\n--- Test Store ---")
    store = create_vector_store(
        test_chunks, test_embeddings, db_path="test_faiss_index"
    )
    if store:
        print("Vector store created.")
        # Test a search
        results = store.similarity_search("What is RAG?")
        print(f"Test search result: {results[0].page_content}")
    print("------------------")
