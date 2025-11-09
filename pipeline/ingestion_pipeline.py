import os
import time

from components.document_loader import load_documents
from components.text_splitter import split_documents
from components.embedding_model import get_embedding_model
from components.vector_store import create_vector_store

from config import DATA_DIRECTORY, VECTOR_DB_PATH, EMBEDDING_MODEL_NAME


def main():
    """
    Runs the full document ingestion pipeline.
    """
    print("Starting document ingestion pipeline...")
    start_time = time.time()

    # --- 1. Load Documents ---
    # Create data directory if it doesn't exist
    if not os.path.exists(DATA_DIRECTORY):
        print(f"Creating data directory: {DATA_DIRECTORY}")
        os.makedirs(DATA_DIRECTORY)
        if not os.listdir(DATA_DIRECTORY):
            with open(os.path.join(DATA_DIRECTORY, "placeholder.txt"), "w") as f:
                f.write(
                    "This is a placeholder file. Add your own documents to the 'data' directory."
                )
                print(
                    "Created placeholder.txt. Please add your own .txt files to the 'data' directory for ingestion."
                )

    documents = load_documents(DATA_DIRECTORY)
    if not documents:
        print("Pipeline stopped: No documents to process.")
        return

    # --- 2. Split Documents ---
    chunks = split_documents(documents)
    if not chunks:
        print("Pipeline stopped: No chunks were created from the documents.")
        return

    # --- 3. Initialize Embedding Model ---
    embeddings = get_embedding_model(EMBEDDING_MODEL_NAME)

    # --- 4. Create and Save Vector Store ---
    print("Creating and saving vector store... This may take a while.")
    vector_store = create_vector_store(chunks, embeddings, VECTOR_DB_PATH)

    if vector_store:
        print("\nPipeline Completed Successfully!")
    else:
        print("\nPipeline FAILED")

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
