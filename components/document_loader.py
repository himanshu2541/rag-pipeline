from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.docstore.document import Document
from typing import Iterator

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
