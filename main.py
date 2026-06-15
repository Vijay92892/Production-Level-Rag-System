import os

from dotenv import load_dotenv
load_dotenv()

from langchain_core import __version__ as core_version
from importlib import metadata

lg_version = metadata.version("langgraph")

from langchain_openai import ChatOpenAI

print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")





def main():
    llm = ChatOpenAI( model="meta/llama-3.3-70b-instruct",
                    base_url="https://integrate.api.nvidia.com/v1",
                        api_key=os.getenv("NVIDIA_API_KEY"),
                        temperature=0.7
                        )
    
    response = llm.invoke("What is the meaning of life?")
    print(response)


if __name__ == "__main__":
    main()
