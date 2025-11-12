from components.format_docs import format_docs
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class ChainProvider:
    """
    Responsible for building the RAG chain.
    """
    def __init__(self):
        self.template = """
        Answer the following question based only on the provided context.
        If you don't know the answer, just say "I do not have enough information to answer."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    
    def get_chain(self, retriever, llm):
        """
        Builds and returns the RAG chain.
        """
        print("Initializing RAG chain...")
        prompt = ChatPromptTemplate.from_template(self.template)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("RAG chain initialized.")
        return rag_chain