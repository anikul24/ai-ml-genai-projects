import os
from token import OP
from typing import Dict, List
import chromadb
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.tools import tool
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv

load_dotenv(dotenv_path="./cred.env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


VECTOR_DB_PATH = "./chroma_db"
COLLECTION_NAME = "retirement_advisor"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo" # or gpt-4o


# 1. Initialize Vector Store (Global connection)
native_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL)


vector_store = Chroma(
    client = native_client,
    collection_name = COLLECTION_NAME,
    embedding_function = embedding_function
)

# 2. Setup Retriever
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 4,
        "score_threshold": 0.4 # Adjust based on your data quality
    }
)


# 3. Setup Memory (Global for the tool)
tool_memory = ConversationBufferWindowMemory(k=5, return_messages=True, memory_key="chat_history", output_key='answer')

# 4. Setup LLM
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

# 5. Setup Chain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=tool_memory,
    return_source_documents=True,
    output_key='answer',
    verbose=True
)

@tool("rag_search")
def rag_search_tool(query:str) -> str:
    """
    Tool that uses RAG to answer retirement-related questions.
    Args:
        query: The user's question (e.g., "What is the WEP penalty?").    
    
    """
    
    print(f"Invoking RAG Chain with query: '{query}'")

    try:

        result = chain.invoke({'question': query})

        answer = result['answer']

        source_docs = result.get('source_documents', [])

        sources_formatted = []
        # Format sources
        if source_docs:
            sources_formatted.append("\nSources:\n")            
            for i, doc in enumerate(source_docs):
                source = doc.metadata.get('source', 'Unknown Source')
                snippet = doc.page_content[:200].replace('\n', ' ') + "..."
                sources_formatted.append (f"[{i+1}] {source}: {snippet}\n")
            
            sources_text = "\n".join(sources_formatted)

            final_response = f"{answer} \n\n Sources:\n{sources_text}"
            return final_response
        
    except Exception as e:
        error_msg = f"Error during RAG search: {e}"
        print(error_msg)
        return error_msg



# --- Test Block ---
if __name__ == "__main__":
    # Simple test to verify it works standalone
    print("Testing Tool...")
    q1 = "How does the Windfall Elimination Provision work?"
    print(rag_search_tool.invoke(q1))
    
    print("\n--- Follow up test (Memory Check) ---")
    q2 = "Does it apply to me if I have 30 years of substantial earnings?"
    print(rag_search_tool.invoke(q2))