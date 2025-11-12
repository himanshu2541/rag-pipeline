import sys
import asyncio

from config import Config
from rag_system import RAGSystem


def print_help():
    print("\n--- RAG Pipeline (Class-Based) ---")
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


async def main_async():
    """
    Asynchronous main function to run the RAG system.
    """
    print("\n===================================")
    print(" Main Application started")
    print("===================================")

    # Initialize the system with the configuration
    try:
        config = Config()
        rag_system = RAGSystem(config)
    except Exception as e:
        print(f"Failed to initialize RAGSystem: {e}")
        return

    if len(sys.argv) < 2:
        print("No command provided.")
        print_help()
        return

    command = sys.argv[1]

    try:
        if command == "--ingest":
            await rag_system.run_ingestion()

        elif command == "--retrieve":
            if len(sys.argv) > 2:
                query_text = " ".join(sys.argv[2:])
                print(f'[main.py] Retrieve Query: "{query_text}"')
                await rag_system.retrieve_chunks(query_text)
            else:
                print("Error: --retrieve command requires text.")
                print('Example: python main.py --retrieve "What is RAG?"')

        elif command == "--ask":
            if len(sys.argv) > 2:
                query_text = " ".join(sys.argv[2:])
                print(f'[main.py] RAG Question: "{query_text}"')
                await rag_system.ask_question(query_text)
            else:
                print("Error: --ask command requires text.")
                print('Example: python main.py --ask "What is RAG?"')

        else:
            print(f"Error: Unknown command '{command}'")
            print_help()
            
    except Exception as e:
        print(f"\n[main.py] An unexpected error occurred: {e}")

    print("\n===================================")
    print(" Main Application Finished")
    print("===================================")


if __name__ == "__main__":
    asyncio.run(main_async())