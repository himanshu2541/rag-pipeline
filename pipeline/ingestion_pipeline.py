import os
import time

from components.document_loader import load_documents_lazy
from components.text_splitter import split_documents
from components.embedding_model import get_embedding_model
from components.vector_store import create_vector_store_from_iterator


from config import DATA_DIRECTORY, VECTOR_DB_PATH, EMBEDDING_MODEL_NAME


def main():
    """
    Runs the full document ingestion pipeline.
    """
    print("Starting document ingestion pipeline...")
    start_time = time.time()

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

    # --- Initialize Components ---
    print("Loading embedding model...")
    embeddings = get_embedding_model(EMBEDDING_MODEL_NAME)

    print("Setting up lazy document loader...")
    document_iterator = load_documents_lazy(DATA_DIRECTORY)

    MODEL_SAFE_CHUNK_SIZE = 1000
    MODEL_SAFE_CHUNK_OVERLAP = 200

    splitter_with_args = lambda docs: split_documents(
        docs, 
        chunk_size=MODEL_SAFE_CHUNK_SIZE, 
        chunk_overlap=MODEL_SAFE_CHUNK_OVERLAP
    )

    # create and store vector store
    vector_store = create_vector_store_from_iterator(
        document_iterator=document_iterator,
        text_splitter_func=splitter_with_args,  # Pass the splitter function
        embedding_model=embeddings,
        db_path=VECTOR_DB_PATH,
        batch_size=100,
    )

    if vector_store:
        print("\nPipeline Completed Successfully!")
    else:
        print("\nPipeline FAILED")

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
