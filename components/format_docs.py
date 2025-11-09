def format_docs(docs):
    """
    Helper runnable to format the retrieved documents for the context.

    Args:
      docs (list): List of document objects with a 'page_content' attribute.

    Returns:
      str: Concatenated string of all document contents separated by double newlines.
    """
    return "\n\n".join(doc.page_content for doc in docs)