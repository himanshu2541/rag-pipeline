from langchain_community.vectorstores import FAISS

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
  

if __name__ == '__main__':
    # simple test
    from components.embedding_model import get_embedding_model
    from langchain_community.docstore.document import Document

    test_chunks = [
        Document(page_content="RAG stands for Retrieval-Augmented Generation.", metadata={"source": "doc1"}),
        Document(page_content="A vector store indexes document embeddings.", metadata={"source": "doc2"})
    ]
    test_embeddings = get_embedding_model()
    
    print("\n--- Test Store ---")
    store = create_vector_store(test_chunks, test_embeddings, db_path="test_faiss_index")
    if store:
        print("Vector store created.")
        # Test a search
        results = store.similarity_search("What is RAG?")
        print(f"Test search result: {results[0].page_content}")
    print("------------------")