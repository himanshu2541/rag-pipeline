import sys
import time
import asyncio

from pipeline.ingestion_pipeline import main as run_ingestion
from pipeline.retrieval_pipeline import query_vector_store as run_retrieval
from pipeline.rag_pipeline import main as run_rag


def print_help():
    print("\n--- RAG Pipeline ---")
    print("Usage: python main.py [command] [options]")
    print("\nCommands:")
    print("  --ingest              Run the document ingestion pipeline.")
    print(
        "                        (Loads docs from './data' and creates 'faiss_index')"
    )
    print('\n  --retrieve "text"     Retrieve relevant chunks from the vector store.')
    print('                        (e.g., python main.py --retrieve "What is RAG?")')
    print(
        '\n  --ask "text"          Ask a question to the full RAG pipeline (Retrieve + Generate).'
    )
    print('                        (e.g., python main.py --ask "What is RAG?")')
    print("\nExample:")
    print("  1. Ingest documents: python main.py --ingest")
    print(
        '  2. Retrieve chunks:  python main.py --retrieve "What is Retrieval-Augmented Generation?"'
    )
    print(
        '  3. Ask question:     python main.py --ask "What is Retrieval-Augmented Generation?"'
    )


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

    elif command == "--retrieve":
        if len(sys.argv) > 2:
            query_text = " ".join(sys.argv[2:])
            print(f'[main.py] Retrieve Query: "{query_text}"')
            print("[main.py] Calling the retrieval-only pipeline module...")
            run_retrieval(query_text)
            print("[main.py] Retrieval-only pipeline module finished.")
        else:
            print("Error: --retrieve command requires text.")
            print('Example: python main.py --retrieve "What is RAG?"')

    elif command == "--ask":
        if len(sys.argv) > 2:
            query_text = " ".join(sys.argv[2:])
            print(f'[main.py] RAG Question: "{query_text}"')
            print("[main.py] Calling the full RAG pipeline (async)...")

            try:
                asyncio.run(run_rag(query_text))
            except Exception as e:
                print(f"\n[main.py] An error occurred during the RAG pipeline: {e}")

            print("[main.py] Full RAG pipeline finished.")
        else:
            print("Error: --ask command requires text.")
            print('Example: python main.py --ask "What is RAG?"')

    else:
        print(f"Error: Unknown command '{command}'")
        print_help()

    end_time = time.time()
    print("\n===================================")
    print(f" Main Application Finished (Total time: {end_time - start_time:.2f}s)")
    print("===================================")


if __name__ == "__main__":
    main_app()
