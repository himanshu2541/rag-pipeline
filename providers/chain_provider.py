from components.format_docs import format_docs
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from typing import cast

class ChainProvider:
    """
    Responsible for building the RAG chain.
    """
    def __init__(self):
        self.rephrase_template = """
        Given the following conversation and a follow-up question, rephrase the 
        follow-up question to be a standalone question, in its original language.

        Chat History:
        {chat_history}

        Follow Up Input: {input}
        Standalone question:"""

        self.answer_template = """
        Answer the following question based only on the provided context.
        If you don't know the answer, just say "I do not have enough information to answer."

        Context:
        {context}

        Question:
        {input}

        Answer:
        """

        
    
    # Local testing purpose
    def get_chain(self, retriever, llm):
        """
        Builds and returns the RAG chain.
        """
        print("Initializing RAG chain...")
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
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("RAG chain initialized.")
        return rag_chain
    
    # conversational chain
    def get_conversational_chain(self, retriever, llm):
        """
        Builds and returns the conversational RAG chain.
        """
        print("Initializing Conversational RAG chain...")

        # 1. Build rephrasing chain
        rephrase_prompt = ChatPromptTemplate.from_template(self.rephrase_template)
        rephrasing_chain = (
            rephrase_prompt
            | llm
            | StrOutputParser()
        )

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, rephrase_prompt
        )

        # 3. Build the Answer-Generation Chain
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", self.answer_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])

        answer_chain = (
            answer_prompt
            | llm
            | StrOutputParser()
        )

        # 4. Combine into Conversational RAG Chain
        conversational_rag_chain = (
            RunnablePassthrough.assign(
                context=history_aware_retriever
            )
            | 
            RunnablePassthrough.assign(context = lambda x: format_docs(x["context"]))
            | answer_chain
        )

        print("Conversational RAG chain initialized.")
        # answer part of the final dictionary
        return conversational_rag_chain