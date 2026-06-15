from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    Language,
)
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import numpy as np
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
load_dotenv()


# Sample documents for testing
SAMPLE_TEXT = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled data to train models. The algorithm learns to map inputs to outputs based on example input-output pairs.

Common algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks

### Unsupervised Learning
Unsupervised learning finds hidden patterns in unlabeled data. The algorithm discovers structure without predefined labels.

Common algorithms include:
- K-Means Clustering
- Principal Component Analysis
- Autoencoders

## Applications

Machine learning is used in many fields:
1. Image recognition
2. Natural language processing
3. Recommendation systems
4. Fraud detection
5. Autonomous vehicles
""".strip()

SAMPLE_CODE = '''
def quicksort(arr):
    """
    Quicksort implementation in Python.
    Time complexity: O(n log n) average, O(n²) worst case.
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)


def binary_search(arr, target):
    """
    Binary search implementation.
    Requires sorted array.
    Time complexity: O(log n)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
'''

def recursive_splitter():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_text(SAMPLE_TEXT)
    
    print(f"Original text length: {len(SAMPLE_TEXT)} chars")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Chunk sizes: {[len(chunk) for chunk in chunks]}")
    print(f"\nFirst chunk preview:\n{chunks[0][:200]}...")



def overlap_importance():
    text = "The quick brown fox jumps over the lazy dog. " * 10  # Repeat to increase length

    #without overlap
    no_overlap = RecursiveCharacterTextSplitter(chunk_size = 50,
                                                chunk_overlap = 0
    )


    with_overlap = RecursiveCharacterTextSplitter(chunk_size = 50,
                                                chunk_overlap = 20
    )

    chunks_no_overlap = no_overlap.split_text(text)
    chunks_with_overlap = with_overlap.split_text(text)

    print("Without overlap:")
    print(f"    Chunk 1 end: ...{chunks_no_overlap[0][-20:]}")
    print(f"    Chunk 1 end: ...{chunks_no_overlap[0][-20:]}...")

    print("\nWith overlap:")
    print(f" Chunk 1 end: ...{chunks_with_overlap[0][-20:]}")
    print(f" Chunk 2 start: ...{chunks_with_overlap[1][:20]}...")

embeddings_model = NVIDIAEmbeddings(
    model="nvidia/llama-nemotron-embed-1b-v2",
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1"
)

def basic_embeddings():
    #single text
    text = "What is Mahine Learning?"
    single_embedding = embeddings_model.embed_query(text)
    print(f"vector dimensions: {len(single_embedding)}")
    print(f"first 5 dimensions: {single_embedding[:5]}")
    print(f"vector norm: {np.linalg.norm(single_embedding):.4f}" )


def batch_embeddings():
    texts = [
        "What is Machine Learning?",
        "How does a neural network work?",
        "What are the applications of AI?"
    ]

    batch_embeddings = embeddings_model.embed_documents(texts)
    print(f"Batch size: {len(batch_embeddings)}")
    print(f"Vector dimensions: {len(batch_embeddings[0])}")
    print(f"First embedding norm: {np.linalg.norm(batch_embeddings[0]):.4f}")


def similarity_search():
    docs = [
        "The cat is on the roof.",
        "The dog is in the yard.",  
        "The bird is in the tree.",
        "The cat is in the yard."
    ]

    query = "Where is the cat?"   

    #embed documents and query
    doc_vector = embeddings_model.embed_documents(docs)
    query_vector = embeddings_model.embed_query(query)

    #compute cosine similarity
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    similarities = [cosine_similarity(query_vector, doc_vec) for doc_vec in doc_vector]

    # rank documents by similarity
    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)

    print("Similarity search results:")
    print("Ranked by similaity")
    for doc, score in ranked_docs:
        print(f"  {score:.4f} {doc}")





if __name__ == "__main__":
    # recursive_splitter()
    # overlap_importance()
    # basic_embeddings()
    # batch_embeddings()
    similarity_search()