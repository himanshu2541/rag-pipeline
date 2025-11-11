import sys
import asyncio
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# uncomment if you want to use local huggingface model
# from langchain_huggingface import HuggingFacePipeline
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
# from langchain_core.runnables import RunnableLambda

from langchain_openai import ChatOpenAI
from components.get_retriever import get_retriever
from components.format_docs import format_docs

# Debug
def log_chain_inputs(input_data):
    """
    A simple pass-through function to print the inputs
    before they go into the prompt template.
    """
    print("\n" + "="*40)
    print("--- DEBUG: Data going into Prompt ---")
    
    # Print the question
    print(f"QUESTION:\n{input_data['question']}")
    
    # Print the context
    print(f"\nCONTEXT:\n{input_data['context']}")
    print("="*40 + "\n")
    
    # Pass the data along
    return input_data

async def main(query_text):
    """
    Runs the full RAG pipeline asynchronously using LCEL (Runnables).
    """
    print("--- Initializing RAG Pipeline ---")
    start_time = time.time()

    # 1. Get the Retriever
    retriever = get_retriever()

    # 2. Get the LLM
    # Using a small llm locally
    # print("Loading LLM (google/flan-t5-small)...")
    # tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    # model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    # hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    # llm = HuggingFacePipeline(pipeline=hf_pipeline)

    llm = ChatOpenAI(
        model="phi-3-mini-4k-instruct",
        base_url="http://localhost:1234/v1",
        api_key=lambda: "type-anything",
    )

    # 3. Define the Prompt
    template = """
    Answer the following question based only on the provided context.
    If you don't know the answer, just say "I do not have enough information to answer."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 4. Define the RAG Chain using Runnables
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        # | RunnableLambda(log_chain_inputs)
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"Pipeline initialized in {time.time() - start_time:.2f}s")

    # 5. Invoke the chain
    print(f"\nQuerying: '{query_text}'")
    start_invoke = time.time()

    print("\nAnswer:")
    full_response = ""
    async for chunk in rag_chain.astream(query_text):
        full_response += chunk
        print(chunk, end="", flush=True)
    print()
    print("--------------------")
    print(f"Query completed in {time.time() - start_invoke:.2f}s")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What is RAG?"

    asyncio.run(main(query))
