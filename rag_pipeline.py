import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough , RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import tempfile

load_dotenv()
embeddings_model = NVIDIAEmbeddings(
    model="nvidia/llama-nemotron-embed-1b-v2",
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1"
)

# Sample knowledge base
KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""
def create_kb():
    """Create a vector store from the knowledge base."""

    #spit the knowledge base into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc = Document(page_content=KNOWLEDGE_BASE, 
                   metadata={"source": "langchain_knowledge_base.md"}
        )
    chunks = splitter.split_documents([doc])

    #create a vector store from the chunks
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=tempfile.mkdtemp()
    )
    return vector_store

def demo_basic_rag():

    vector_store = create_kb()
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    llm = ChatOpenAI( model="meta/llama-3.3-70b-instruct",
                    base_url="https://integrate.api.nvidia.com/v1",
                        api_key=os.getenv("NVIDIA_API_KEY"),
                        temperature=0.2
                 )
    
    #RAG prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based only on the following context.
        {context}

        Question: {question}
        Answer:

        Make sure to answer in a concise manner,
        and if you don't know the answer, say you don't know."""
    )

    #Format retriever results docs
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    #Rag chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    #Test the RAG chain
    questions = [
        "What is LangChain?",
        "What are the core components of LangChain?",
        "Who created LangChain and when?"
        ]
    
    print("Basic RAG Demo:\n")
    for question in questions:
        answer = rag_chain.invoke(question)
        print(f"Q: {question}\nA: {answer}\n")

if __name__ == "__main__":
    demo_basic_rag()