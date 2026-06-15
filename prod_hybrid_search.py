import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()

# embeddings_model = NVIDIAEmbeddings(
#     model="nvidia/llama-nemotron-embed-1b-v2",
#     api_key=os.getenv("NVIDIA_API_KEY"),
#     base_url="https://integrate.api.nvidia.com/v1"
# )




documents = [
    Document(
        page_content='Product SKU-7742X is our flagship router. It supports '
                     'gigabit speeds and advanced QoS features.',
        metadata={'type': 'product'}
    ),
    Document(
        page_content='For network connectivity issues, first check the '
                     'ethernet cable and router status lights.',
        metadata={'type': 'troubleshooting'}
    ),
    Document(
        page_content='Error code E_CONN_REFUSED indicates the server '
                     'rejected the connection. Check firewall settings.',
        metadata={'type': 'error'}
    ),
    Document(
        page_content='The authentication process requires valid credentials. '
                     'Use OAuth2 for secure API access.',
        metadata={'type': 'auth'}
    ),
    Document(
        page_content='Router configuration guide: Access the admin panel '
                     'at 192.168.1.1 to modify settings.',
        metadata={'type': 'config'}
    ),
    Document(
        page_content='WCAG 2.1 compliance requires all images to have '
                     'alt text and sufficient color contrast.',
        metadata={'type': 'compliance'}
    ),
]



print(f'Loaded {len(documents)} documents')

# create embeddings and vector store


embeddings = NVIDIAEmbeddings(
    model="nvidia/llama-nemotron-embed-1b-v2",
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1"
)

vectorstore = Chroma.from_documents(
    documents,
    embeddings,
    collection_name="hybrid_test"
)

#create vector retriever
vector_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

print('Vector retriever ready')

#bm25 retriever on the raw text
bm25_retriever = BM25Retriever.from_documents(
    documents,
    k=3
)

print('BM25 retriever ready')

#combine them in an ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)

print('Hybrid retriever ready')

def test_retriever(query, name, retriever):
    '''Test a query and show results'''
    results = retriever.invoke(query)
    print(f'\n{name} results for query: "{query}"')
    for i, doc in enumerate(results[:3]):
        preview = doc.page_content[:80] + '...'
        print(f'{i+1}. {preview}')
    return results


test_queries = [
    'SKU-7742X specifications',    # Exact product code
    'E_CONN_REFUSED error',         # Error code
    'How do I authenticate?',       # Semantic question
    'WCAG compliance',              # Acronym
    'router configuration',         # General semantic
]

for query in test_queries:
    print('=' * 60)

    #vector only
    vector_results = test_retriever(query, 'Vector', vector_retriever)

    #bm25 only
    bm25_results = test_retriever(query, 'BM25', bm25_retriever)

    #ensemble
    ensemble_results = test_retriever(query, 'Hybrid', ensemble_retriever)