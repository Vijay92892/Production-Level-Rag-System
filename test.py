from dotenv import load_dotenv
load_dotenv()
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import os 

# print(os.getenv("NVIDIA_API_KEY"))

embeddings = NVIDIAEmbeddings(
    model="nvidia/llama-nemotron-embed-1b-v2",
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1"
)

vector = embeddings.embed_query(
    "What is retrieval augmented generation?"
)

print(len(vector))
print(vector[:5])