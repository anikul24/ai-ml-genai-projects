import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from dotenv import load_dotenv
import chromadb
from tqdm import tqdm
import pytesseract
#from langchain_community.document_loaders import PyPDFLoader,UnstructuredURLLoader, UnstructuredFileLoader
from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores import Chroma # or Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# CONFIG
DATA_DIR = Path("data/raw/ssa/pdf/")
VECTOR_DB_PATH = "./chroma_db"
COLLECTION_NAME = "retirement_advisor"

# This tells Python: "Don't look in the PATH, look exactly here."
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

load_dotenv(dotenv_path="./cred.env")


# Pinecone config from env
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

VECTOR_DB_TYPE = 'CHROMA' # 'PINECONE'

VECTOR_INDEX = os.environ.get("VECTOR_INDEX", "retirement-ssa-index")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# Embedding model name
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")



def load_pdfs_FileLoader(directory) -> List[Document]:
    """Loads all PDFs from the directory using UnstructuredLoader"""
    documents = []

    pdf_files = list(directory.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {directory}")
        return []


    for file_path in tqdm(pdf_files, desc="Loading PDFs", unit="file"):
        print(f"Loading: {file_path.name}")
        try:
            loader = UnstructuredLoader(
                            file_path=str(file_path),
                            strategy="hi_res",
                            mode="elements",  # Important: returns specific chunks/elements instead of one giant text
                            infer_table_structure=True, # Correct param name for table detection
                            chunking_strategy="by_title" # Uses document structure (headers) to chunk
                        )
            docs = loader.load()
            # Add metadata for citation
            for doc in docs:
                doc.metadata["source"] = file_path.name
                doc.metadata["type"] = "SSA_official_publication"
                doc.metadata.pop("languages", None)
            documents.extend(docs)
        except Exception as e:
            tqdm.write(f"Failed to load {file_path.name}: {e}")
    return documents

def main():
    # 0. Setup Chroma Client to check existing data
    native_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

    # Try to get the collection to check its count
    try:
        collection = native_client.get_collection(name=COLLECTION_NAME)
        doc_count = collection.count()
    except ValueError:
        # Collection doesn't exist yet
        doc_count = 0
        

    print(f"Current collection '{COLLECTION_NAME}' has {doc_count} documents.")

    # --- LOGIC TO SKIP LOADING ---
    if doc_count > 0:
        print(" Database already populated. Skipping PDF loading and ingestion.")
        print("To re-ingest, please delete the './chroma_data' folder or rename the collection.")
        return # EXIT FUNCTION EARLY
    # -----------------------------

    # 1. Load PDFs
    if not DATA_DIR.exists():
        print(f"Error: Directory {DATA_DIR} does not exist.")
        return

    raw_docs = load_pdfs_FileLoader(DATA_DIR)
    print(f"Loaded {len(raw_docs)} pages from PDFs.")

    # 2. Chunking (PDFs need good chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=500,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_docs)
    print(f"Created {len(chunks)} chunks.")

    # 3. Embed & Upsert (ChromaDB Example)
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_DB_PATH
    )
    print("Ingestion Complete!")

if __name__ == "__main__":
    main()