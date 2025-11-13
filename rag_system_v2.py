import time
import asyncio
from typing import List, Tuple

from config import Config

from components.embedding_model import get_embedding_model

from components.text_splitter import split_documents
from langchain_community.docstore.document import Document
from langchain_core.messages import HumanMessage, AIMessage

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
        
        # 2. Initialize the Ingestor (for --ingest command, if you keep it)
        self.ingestor = Ingestor(self.config, self.embeddings)
        
        # 3. Initialize Providers
        self.llm_provider = LLMProvider(self.config)
        self.chain_provider = ChainProvider()
        
        # 4. Initialize the retriever provider
        self.retriever_provider = RetrieverProvider(self.config, self.embeddings)

        # 5. Lazy-loaded components
        self._retriever = None
        self._rag_chain = None
        print("RAG System initialized.")

    async def run_ingestion(self):
        """
        Runs the ingestion pipeline using the Ingestor component
        (e.g., for initial data directory setup).
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.ingestor.run)
        print("Ingestion complete. Vector store should now be ready.")
        self._retriever = None 
        self._rag_chain = None

    def _get_retriever(self):
        """Lazy-loads the retriever on first access."""
        if self._retriever is None:
            print("Loading retriever...")
            self._retriever = self.retriever_provider.get_retriever()
        return self._retriever

    def _get_rag_chain(self):
        """Lazy-loads the RAG chain on first access."""
        if self._rag_chain is None:
            print("Loading RAG chain...")
            retriever = self._get_retriever()
            llm = self.llm_provider.get_llm()
            self._rag_chain = self.chain_provider.get_conversational_chain(retriever, llm)
        return self._rag_chain

    async def add_document_from_text(self, text_content: str, source_name: str = "uploaded_file"):
        """
        Ingests a single document from a text string.
        This will update FAISS and trigger a full rebuild of the BM25 index.
        """
        print(f"Ingesting new document: {source_name}")
        
        # 1. Create a Document object
        doc = Document(page_content=text_content, metadata={"source": source_name})
        
        # 2. Split the document
        chunks = split_documents(
            [doc],
            chunk_size=self.config.MODEL_SAFE_CHUNK_SIZE,
            chunk_overlap=self.config.MODEL_SAFE_CHUNK_OVERLAP,
        )
        
        if not chunks:
            print("Warning: No chunks created from the document.")
            return

        print(f"Created {len(chunks)} chunks for {source_name}")

        try:
            loop = asyncio.get_event_loop()
            
            # 3. Get the FAISS vector store
            print("Getting vector store for update...")
            vector_store = await loop.run_in_executor(
                None, self.retriever_provider.get_vector_store
            )

            # 4. Add documents to the FAISS vector store
            await vector_store.aadd_documents(chunks)
            print("Chunks added to FAISS store.")

            # 5. Save the updated FAISS index (synchronous)
            await loop.run_in_executor(
                None, 
                vector_store.save_local, 
                self.config.VECTOR_DB_PATH
            )
            print(f"FAISS index saved to {self.config.VECTOR_DB_PATH}")

            # 6. CRITICAL: Rebuild the BM25 index (synchronous)
            # This is slow but necessary for the ensemble retriever to work.
            print("Rebuilding BM25 index. This may take a moment...")
            await loop.run_in_executor(
                None, self.retriever_provider._build_and_save_bm25, vector_store
            )
            print("BM25 index has been rebuilt and saved.")

            # 7. Force retriever and chain to reload on next call
            # This ensures they pick up the newly indexed data.
            self._retriever = None
            self._rag_chain = None

        except Exception as e:
            print(f"Error adding document to vector store: {e}")


    async def retrieve_chunks(self, query_text: str):
        """
        Retrieves and prints relevant chunks for a given query.
        (No history used here, just simple retrieval)
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

    async def answer_question(self, query_text: str, chat_history: List[Tuple[str, str]] = []):
        """
        Asks a question to the full RAG pipeline and streams the answer.
        """
        print(f"\n--- Querying RAG Pipeline: '{query_text}' ---")
        start_time = time.perf_counter()

        try:
            rag_chain = self._get_rag_chain()
            
            history_messages = []
            for human, ai in chat_history:
                history_messages.append(HumanMessage(content=human))
                history_messages.append(AIMessage(content=ai))

            print("\nAnswer:")
            full_response = ""
            
            input_dict = {
                "input": query_text,
                "chat_history": history_messages
            }
            
            async for chunk in rag_chain.astream(input_dict):
                full_response += chunk
                print(chunk, end="", flush=True)
            print()
            
            end_time = time.perf_counter()
            print("--------------------")
            print(f"Query completed in {end_time - start_time:.2f}s")
            
            return full_response
            
        except Exception as e:
            print(f"\n[RAGSystem] An error occurred during the RAG pipeline: {e}")
            return "An error occurred while processing your request."