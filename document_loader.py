import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import ( TextLoader, 
                                                  WebBaseLoader,
                                                  PyPDFLoader,
                                                  DirectoryLoader
                                                   )


from dotenv import load_dotenv
load_dotenv()

def loadtextfile() :
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(b"This is a sample text file for testing.\nThis is used to test the TextLoader in langchain.")
        tmp_file_path = tmp_file.name
    
    try:
        loader = TextLoader(tmp_file_path)
        documents = loader.load()
        
        print(f"Loaded {len(documents)} document(s) ")
        print(f"Document content: {documents[0].page_content[:100]}...")
        print(f"Document metadata: {documents[0].metadata}")

        # for doc in documents:
        #     print("Document cotent:")
        #     print(doc)
        #     print(doc.page_content)

    finally:    
        os.remove(tmp_file_path)








def pdf_loader(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print(f"Loaded {len(documents)} document(s) from PDF")
    for i, doc in enumerate(documents):
        print(f"Document {i+1} content: Preview: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")






if __name__ == "__main__":
    # loadtextfile()
    pdf_loader("./docs/langchain_demo.pdf")