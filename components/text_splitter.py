from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
  """
  Splits a list of documents into smaller chunks.

  Args:
    documents (list): The list of Document objects to split.
    chunk_size (int): The maximum size of each chunk (in characters).
    chunk_overlap (int): The number of characters to overlap between chunks.

  Returns:
    list: A list of smaller Document objects (chunks).
  """

  print(f"Splitting {len(documents)} document(s) into chunks (size={chunk_size}, overlap={chunk_overlap})....")
  
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = chunk_size,
    chunk_overlap = chunk_overlap,
    length_function = len
  )

  chunks = text_splitter.split_documents(documents)

  print(f"Successfully split into {len(chunks)} chunks.")
  return chunks