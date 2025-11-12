import time
import asyncio

from config import Config

from components.embedding_model import get_embedding_model

from providers.ingestor import Ingestor
from providers.llm_provider import LLMProvider
from providers.retriever_provider import RetrieverProvider
from providers.chain_provider import ChainProvider

class RAGSystem:
    """
    RAG Pipline: Ingestion, Retrieval, and Generation System.
    """

    def __init__(self, config: Config):
        self.config = config
        
        print("Initializing RAG System...")
        # 1. Load embeddings
        self.embeddings = get_embedding_model(self.config.EMBEDDING_MODEL_NAME)
        
        # 2. Initialize the Ingestor
        self.ingestor = Ingestor(self.config, self.embeddings)
        
        # 3. Initialize Providers
        self.llm_provider = LLMProvider(self.config)
        self.chain_provider = ChainProvider()
        # Defer retriever provider initialization until needed, as it checks for
        # the vector store, which might not exist before ingestion.
        
        # 4. Lazy-loaded components
        self._retriever = None
        self._rag_chain = None
        print("RAG System initialized.")

    async def run_ingestion(self):
        """
        Runs the ingestion pipeline using the Ingestor component.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.ingestor.run)
        print("Ingestion complete. Vector store should now be ready.")

    def _get_retriever(self):
        """Lazy-loads the retriever on first access."""
        if self._retriever is None:
            retriever_provider = RetrieverProvider(self.config, self.embeddings)
            self._retriever = retriever_provider.get_retriever()
        return self._retriever

    def _get_rag_chain(self):
        """Lazy-loads the RAG chain on first access."""
        if self._rag_chain is None:
            retriever = self._get_retriever()
            llm = self.llm_provider.get_llm()
            self._rag_chain = self.chain_provider.get_chain(retriever, llm)
        return self._rag_chain

    async def retrieve_chunks(self, query_text: str):
        """
        Retrieves and prints relevant chunks for a given query.
        """
        print(f"\n--- Starting Retrieval for: '{query_text}' ---")
        start_time = time.perf_counter()
        
        try:
            retriever = self._get_retriever()
            results = await retriever.ainvoke(query_text)
            
            end_time = time.perf_counter()
            print(f"Query completed in {end_time - start_time:.2f}s")

            if not results:
                print("No results found.")
                return

            print(f"\nFound {len(results)} relevant chunks:")
            for i, doc in enumerate(results):
                print("---------------------------------")
                print(f"Result {i+1} (from: {doc.metadata.get('source', 'Unknown')}):")
                print(doc.page_content)
                print("---------------------------------")
        
        except Exception as e:
            print(f"Error during retrieval: {e}")

    async def ask_question(self, query_text: str):
        """
        Asks a question to the full RAG pipeline and streams the answer.
        """
        print(f"\n--- Querying RAG Pipeline: '{query_text}' ---")
        start_time = time.perf_counter()

        try:
            rag_chain = self._get_rag_chain()
            
            print("\nAnswer:")
            full_response = ""
            async for chunk in rag_chain.astream(query_text):
                full_response += chunk
                print(chunk, end="", flush=True)
            print()
            
            end_time = time.perf_counter()
            print("--------------------")
            print(f"Query completed in {end_time - start_time:.2f}s")
            
        except Exception as e:
            print(f"\n[RAGSystem] An error occurred during the RAG pipeline: {e}")