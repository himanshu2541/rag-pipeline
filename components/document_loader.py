import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.docstore.document import Document
from typing import Iterator


def load_documents(directory_path="./data"):
    """
    Loads all documents from the specified directory.

    For now, this only take .txt files, but this can be easily expanded.

    Args:
        directory_path (str): The path to the directory containing documents.

    Returns:
        list: A list of Document objects loaded by Langchain
    """

    print(f"Loading documents from: {directory_path}")

    # using TextLoader for .txt files.
    # Use PyPDFLoader for .pdf, etc.
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.txt",  # Load only .txt files
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "UTF-8"},
        show_progress=True,
        use_multithreading=True,
    )

    try:
        documents = loader.load()
        if not documents:
            print(
                f"Warning: No documents found in {directory_path}. Make sure you have .txt files in there."
            )
            return []
        print(f"Successfully loaded {len(documents)} document(s).")
        return documents
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []


# ADDING NEW FUNCTION TO DO BATCH PROCESSING
def load_documents_lazy(directory_path="./data") -> Iterator[Document]:
    """
    Lazily loads document from the specified directory.
    This returns an iterator.

    Args:
        directory_path (str): The path to the directory containing documents.

    Returns:
        (Iterator[Document]): Iterator yielding Document objects.
    """
    print(f"Setting up lazy loader from: {directory_path}")

    loader = DirectoryLoader(
        directory_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "UTF-8"},
        show_progress=True,
        use_multithreading=True,
    )

    try:
        return loader.lazy_load()
    except Exception as e:
        print(f"Error setting up lazy loader: {e}")
        return iter([])  # returns empty iterator


if __name__ == "__main__":
    # Simple test to run this module directly
    # Create a './data' directory and put .txt file in it to test

    if not os.path.exists("./data"):
        os.makedirs("./data")
        with open("./data/test.txt", "w") as f:
            f.write("This is a test document.")

    docs = load_documents()
    if docs:
        print("\n---- Test Load ----")
        print(f"Loaded {len(docs)} doc(s).")
        print(f"First doc content: {docs[0].page_content[:50]}...")
        print(f"First doc metadata: {docs[0].metadata}")
        print("------------------------")
