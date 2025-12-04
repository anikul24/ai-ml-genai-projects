from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import chromadb
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path="./cred.env")

# Configuration (Must match your ingestion script)
VECTOR_DB_PATH = "./chroma_db"
COLLECTION_NAME = "retirement_advisor"
EMBEDDING_MODEL = "text-embedding-3-small"

def verify_chroma():
    print(f"🔍 Checking ChromaDB at: {VECTOR_DB_PATH}")
    
    # 1. Check Document Count (using native client for speed)
    try:
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        count = collection.count()
        print(f"Total Documents Indexed: {count}")
        
        if count == 0:
            print("Collection exists but is empty!")
            return
    except Exception as e:
        print(f"Error accessing collection: {e}")
        return

    # 2. Test Semantic Search (using LangChain wrapper)
    print("\nTesting Semantic Search...")
    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )
        
        # Ask a question relevant to your PDFs (e.g., about Windfall Elimination)
        query = "How does the Windfall Elimination Provision affect my pension?"
        results = vector_store.similarity_search(query, k=2)
        
        print(f"\nQuery: '{query}'")
        print(f"Found {len(results)} relevant matches:\n")
        
        for i, doc in enumerate(results):
            print(f"--- Match {i+1} ---")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Content Snippet: {doc.page_content[:600]}...\n")
            
    except Exception as e:
        print(f" Search failed: {e}")

if __name__ == "__main__":
    verify_chroma()