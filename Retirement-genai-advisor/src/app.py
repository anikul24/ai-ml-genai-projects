import streamlit as st
import os
from typing import List
from langchain_core.messages import HumanMessage, AIMessage

##include custom components
from graph import create_retirement_graph
from tools_rag import rag_search_tool
from tools_math import calculate_retirement_growth, calculate_rmd
from tools_web import web_search
from tools_user import create_user_doc_tool

st.set_page_config(page_title="Retirement GenAI Advisor", layout="wide")
st.title("GenAI Retirement Advisor")


# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_tools" not in st.session_state:
    st.session_state.user_tools = []

if "graph_app" not in st.session_state:
    # Initial Build: Standard Tools Only
    default_tools = [rag_search_tool, calculate_retirement_growth, calculate_rmd, web_search]
    st.session_state.graph_app = create_retirement_graph(default_tools)

# --- SIDEBAR: DOCUMENT UPLOAD ---
with st.sidebar:
    st.header("📂 Analyze My Documents")
    st.write("Upload a 401(k) statement or pension doc to ask specific questions.")
    
    # Updated line to accept both PDF and XML
    uploaded_file = st.file_uploader("Upload 401k/Pension Doc", type=["pdf", "xml"])

    #if file uploaded and NOT processed yet
    if uploaded_file and not st.session_state.get("file_processed"):
        with st.spinner("Processing document..."):
            # 1. Save to temp file
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 2. Create the Custom Tool
            # We use 'session_id' to keep vector stores separate if needed
            new_tool = create_user_doc_tool(temp_path, session_id="current_user")
            
            if new_tool:
                st.session_state.user_tools = [new_tool]
                
                # 3. REBUILD THE GRAPH with the new tool included
                all_tools = [rag_search_tool, calculate_retirement_growth, calculate_rmd, web_search, new_tool]
                st.session_state.graph_app = create_retirement_graph(all_tools)
                
                st.session_state["file_processed"] = True
                st.success(f"Loaded {uploaded_file.name}! You can now ask questions about it.")
            else:
                st.error("Could not process file.")


persona = st.selectbox("Adjust Persona", ["retiree", "financial_planner", "family_member"])
st.session_state["user_persona"] = persona

# --- CHAT INTERFACE ---

# 1. Display History
for msg in st.session_state.messages:
    if msg.type == "human":
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif msg.type == "ai":
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# 2. Chat Input
if user_input := st.chat_input("Ask about retirement, taxes, or your uploaded docs..."):
    # Display User Message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    # 3. Run Agent
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Prepare State
        initial_state = {
            "messages": st.session_state.messages,
            "user_persona": persona
        }

        # Stream the graph events
        # We use 'stream' to see thoughts (optional) or just get final answer
        app = st.session_state.graph_app

        try:
            # Stream output to show "thinking"
            for event in app.stream(initial_state):
                # Inspect event to see if it's a tool call or final answer
                for key, value in event.items():
                    if "messages" in value:
                        last_msg = value["messages"][-1]
                        if isinstance(last_msg, AIMessage) and last_msg.content:
                            full_response = last_msg.content
                            message_placeholder.markdown(full_response)
            # Save AI response to history
            st.session_state.messages.append(AIMessage(content=full_response))
        except Exception as e:
            error_msg = f"Error during response generation: {e}"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append(AIMessage(content=error_msg))
                                                               