import os
from dotenv import load_dotenv
load_dotenv()

from  openai import OpenAI

client = OpenAI(
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1",
)

response = client.embeddings.create(
    input="Hello, world!",
    model="nvidia/llama-nemotron-embed-1b-v2",
    extra_body = {
        "input_type": "query",
    }
)
print(response)
print(len(response.data[0].embedding))



