import os
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import Tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from dotenv import load_dotenv

load_dotenv(dotenv_path="./cred.env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


VECTOR_DB_PATH = "./chroma_db"
COLLECTION_NAME = "retirement_advisor"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo" # or gpt-4o


def processing_uploaded_file(file_path: str) -> List[str]:
    """
        Processes the uploaded file and returns text chunks.
    """
    try:

        loader = UnstructuredFileLoader(
            file_path=file_path,
            strategy="fast",
            mode="elements",
        )

        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        return chunks

    except Exception as e:
        print(f"Error processing file: {e}")
        return None





def create_user_doc_tool(file_path: str, session_id: str):
    """
        Creates a TEMPORARY retrieval tool for the specific file uploaded by the user.
    """

    #1.load and chunk the file
    chunks = processing_uploaded_file(file_path)
    if not chunks:
        return None
    

    #2.create vector store
    #unique collection name per session so data doesn't leak
    temp_collection_name = f"user_docs_{session_id}"

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    ##crete vector store from chunks
    vector_store  = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=temp_collection_name
    )

    # 3. Create the Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})