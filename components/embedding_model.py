from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model(model_name="all-MiniLm-L6-v2", device="cpu"):
    """
    Initializes and returns the embedding model.

    We are using local, open-source Hugging Face model so we don't need an API key for now

    Args:
      model_name (str): The name of the HuggingFace model to use.
      device (str): The device to run the model on ('cpu' or 'cuda')

    Returns:
      HuggingFaceEmbeddings: The initialized embedding model object.
    """

    print(f"Initializing embedding model: {model_name} on device: {device}")

    model_kwargs = {'device': device}

    # For sentence-transformers, encode kwargs specifies normalization
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name = model_name,
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs
    )

    print("Embedding model initialized successfully.")
    return embeddings

if __name__ == '__main__':
    # simple test 
    embeddings = get_embedding_model()
    
    print("\n--- Test Embed ---")
    test_vector = embeddings.embed_query("This is a test sentence.")
    print(f"Embedding model is working.")
    print(f"Test vector length: {len(test_vector)}")
    print("------------------")