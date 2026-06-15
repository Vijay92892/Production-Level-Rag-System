import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_core.documents import Document


load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_DATABASE_URL")

#For demo use local if Supabase is not configured
DATABASE_URL = SUPABASE_URL or os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/postgres"
)


def connect_to_supabase():
    """Connec to supabase pgvector store"""

    embeddings = NVIDIAEmbeddings(
    model="nvidia/llama-nemotron-embed-1b-v2",
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1"
    )

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name = "production_docs",
        connection = DATABASE_URL,
        use_jsonb = True
    )

    return vectorstore

def verify_connection(vectorstore):
    """Verify connection by adding and retrieving a test document"""

    test_doc = Document(
        page_content = "This is a test document to verify Supabase",
        metadata = {"test": True}
    )

    try:
        ids = vectorstore.add_documents([test_doc])
        print(f"Test document added with ID: {ids[0]}")

        #search for it
        results = vectorstore.similarity_search("test document", k=1)
        if results:
            print(f"Search  works:{results[0].page_content}")
        
        #cleanup
        vectorstore.delete(ids)
        print("Cleanup complete")

        return True
    
    except Exception as e:
        print(f"Error : {e}")
        return False
    
def main():

    print("=" * 60)
    print("Supabase PGVector Connection Test")
    print("=" * 60)

    if SUPABASE_URL:
        print("\nConnecting to Supabase...")
        print(f"Host: {SUPABASE_URL.split('@')[1].split('/')[0]}")
    else:
        print("\nSUPABASE_DATABASE_URL not set.")
        print("Using local PostgreSQL instead.")

    try:
        vectorstore = connect_to_supabase()

        success = verify_connection(vectorstore)

        if success:
            print("\n✅ Connection verified successfully!")
        else:
            print("\n❌ Connection verification failed.")

    except Exception as e:
        print(f"\n❌ Failed to connect: {e}")

if __name__ == "__main__":
    main()