import chromadb
chroma_client = chromadb.Client()

collection_name = "test_collection"

collection = chroma_client.get_or_create_collection(name=collection_name)

documents = [
    {"id" : "doc1", "text": "Hello, World!"},
    {"id" : "doc2", "text": "How are you today?"},
    {"id" : "doc3", "text": "Goodbye, See you later!"}
]

for doc in documents:
    collection.upsert(ids = doc["id"], documents = doc["text"])

query = "Hello, World!"

results = collection.query(query_texts=[query], n_results=2)

print(results)