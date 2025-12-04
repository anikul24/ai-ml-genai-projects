import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from tqdm import tqdm
import pytesseract
import chromadb 
from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- CONFIGURATION ---
# Map folder names to specific metadata types
# The script will use these types to check if data is already indexed.
SOURCE_MAP = {
    "ssa": "SSA_official_publication",
    "irs": "IRS_tax_rule",
    "medicare": "Medicare_guide",
    "providers": "Investment_provider_doc",
    "finance_101": "General_retirement_guide"
}

BASE_DATA_DIR = Path("data/raw")
VECTOR_DB_PATH = "./chroma_db"
COLLECTION_NAME = "retirement_advisor"
EMBEDDING_MODEL = "text-embedding-3-small"

# Explicit Tesseract Path (Ensure this matches your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

load_dotenv(dotenv_path="./cred.env")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def load_documents_from_folder(directory: Path, doc_type: str) -> List[Document]:
    """Loads PDF, DOCX, TXT, XLSX from a specific folder."""
    documents = []
    
    # Look for common document extensions
    files = list(directory.glob("*.*")) 
    valid_files = [f for f in files if f.suffix.lower() in ['.pdf', '.docx', '.txt', '.xlsx']]

    if not valid_files:
        print(f"  - No valid documents found in {directory}")
        return []

    print(f"  - Found {len(valid_files)} files in '{directory.name}' (Type: {doc_type})")

    for file_path in tqdm(valid_files, desc=f"Loading {directory.name}", unit="file"):
        try:
            # Intead of Using hi_res strategy , used fast strategy for faster pdf reading
            loader = UnstructuredLoader(
                file_path=str(file_path),
                strategy="fast", ## chanfing from 'hi_res' to fast because of perfromance
                mode="elements",
                infer_table_structure=True,
                chunking_strategy="by_title"
            )
            docs = loader.load()
            
            # Add Source-Specific Metadata
            for doc in docs:
                doc.metadata["source"] = file_path.name
                doc.metadata["type"] = doc_type
                
                # Clean up complex metadata that might break Chroma (like lists)
                for key, value in list(doc.metadata.items()):
                    if isinstance(value, list):
                        doc.metadata[key] = ", ".join(map(str, value))
            
            documents.extend(docs)
        except Exception as e:
            # Using tqdm.write to print errors without breaking the progress bar
            tqdm.write(f"    [Error] Failed to load {file_path.name}: {e}")
            
    return documents

def main():
    # 1. Initialize Native Client to check for existing data
    native_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    collection = None
    
    try:
        collection = native_client.get_collection(name=COLLECTION_NAME)
        print(f"Accessing existing collection: '{COLLECTION_NAME}'")
    except ValueError:
        print(f"Collection '{COLLECTION_NAME}' does not exist yet. It will be created.")
        # Collection is None, so we will process everything

    all_chunks = []
    
    # 2. Iterate through our defined sources
    print(f"Starting Ingestion from {BASE_DATA_DIR.resolve()}")
    
    for folder_name, doc_type in SOURCE_MAP.items():
        folder_path = BASE_DATA_DIR / folder_name
        
        # SKIP LOGIC: Check if this specific doc_type is already in the DB
        if collection:
            existing_docs = collection.get(where={"type": doc_type}, limit=1)
            if existing_docs['ids']:
                print(f"[Skip] Folder '{folder_name}' (Type: {doc_type}) is already indexed.")
                continue

        # If not skipped, proceed to load
        if folder_path.exists():
            print(f"\nProcessing folder: {folder_name}...")
            raw_docs = load_documents_from_folder(folder_path, doc_type)
            
            if raw_docs:
                # Chunk immediately to manage memory per folder
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=4000,
                    chunk_overlap=500,
                    separators=["\n\n", "\n", " ", ""]
                )
                folder_chunks = text_splitter.split_documents(raw_docs)
                all_chunks.extend(folder_chunks)
                print(f"    -> Generated {len(folder_chunks)} chunks from {folder_name}")
        else:
            # This is just info, not an error, as you might not have all folders yet
            pass

    if not all_chunks:
        print("No new documents to process. Exiting.")
        return

    # 3. Embed & Upsert All New Chunks
    print(f"\nUpserting {len(all_chunks)} total new chunks to ChromaDB...")
    
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Chroma.from_documents handles batching and creation automatically
    vector_store = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_DB_PATH
    )
    print("Ingestion Complete.")

if __name__ == "__main__":
    main()