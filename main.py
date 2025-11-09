import sys
import time

from pipeline.ingestion_pipeline import main as run_ingestion
from pipeline.retrieval_pipeline import query_vector_store

def print_help():
    print("\n--- RAG Pipeline ---")
    print("Usage: python main.py [command] [options]")
    print("\nCommands:")
    print("  --ingest          Run the document ingestion pipeline.")
    print("                    (Loads docs from './data' and creates 'faiss_index')")
    print("\n  --query \"text\"    Query the existing vector store with your text.")
    print("                    (e.g., python main.py --query \"What is RAG?\")")
    print("\nExample:")
    print("  1. Ingest documents: python main.py --ingest")
    print("  2. Query documents:  python main.py --query \"What is Retrieval-Augmented Generation?\"")

def main_app():
    print("\n===================================")
    print(f" Main Application started")
    print("===================================")
    start_time = time.time()
    
    if len(sys.argv) < 2:
        print("No command provided.")
        print_help()
        return

    command = sys.argv[1]

    if command == "--ingest":
        print("[main.py] Calling the ingestion pipeline module...")
        run_ingestion()
        print("[main.py] Ingestion pipeline module finished.")
        
    elif command == "--query":
        if len(sys.argv) > 2:
            query_text = " ".join(sys.argv[2:])
            print(f"[main.py] Query: \"{query_text}\"")
            print("[main.py] Calling the retrieval pipeline module...")
            query_vector_store(query_text)
            print("[main.py] Retrieval pipeline module finished.")
        else:
            print("Error: --query command requires text.")
            print("Example: python main.py --query \"What is RAG?\"")
            
    else:
        print(f"Error: Unknown command '{command}'")
        print_help()
    
    
    end_time = time.time()
    print("\n===================================")
    print(f" Main Application Finished (Total time: {end_time - start_time:.2f}s)")
    print("===================================")


if __name__ == "__main__":
    main_app()

