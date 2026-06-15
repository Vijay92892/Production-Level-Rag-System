"""
Langsmith Setup and Observability
Production monitiring for Langchain/Langgraph
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from langsmith.run_trees import RunTree
from dotenv import load_dotenv
# from langsmith import Client

# client = Client()
# print(client)

load_dotenv()

# Enable tracing 
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "multi-agent-research-system"


@traceable(name = "basic_chaining", tags= ["production", "summarization"])
def demo_basic_tracing():
    """ Basic langsmith tracing"""
    
    llm = ChatOpenAI( model="meta/llama-3.3-70b-instruct",
                    base_url="https://integrate.api.nvidia.com/v1",
                        api_key=os.getenv("NVIDIA_API_KEY"),
                        temperature=0.7
                        )
    
    prompt = ChatPromptTemplate.from_template(
        " Explain {topic} in one sentence."
    )

    chain = prompt | llm | StrOutputParser()

    print("Basic Tracing Demo:\n")
    print("Running chain with LangSmith tracing enabled...")

    result = chain.invoke({"topic": "machine learning"})

    print(f"Result: {result}")
    print("\nCheck LangSmith dashboard for trace details.")


if __name__ == "__main__":
    demo_basic_tracing()