import os
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from functools import partial

#from IPython.display import Image, display

# Import custom components
from tools_rag import rag_search_tool
from tools_math import calculate_retirement_growth, calculate_rmd
from personas import get_system_prompt
from state import AgentState
from tools_web import web_search



##tools_list

tools = [
    rag_search_tool, 
    calculate_retirement_growth, 
    calculate_rmd, 
    web_search 
]


# Initialize the LLM and bind tools

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)



# Create Nodes

def orchestrator_node(state:AgentState, llm_with_tools:ChatOpenAI):
    """
        The 'Brain' node. It decides whether to speak to the user or call a tool. 
        It is the Orchestrator Node that decides which tool to use based on user query and state.
    """

    #get persona
    persona = state.get("user_persona", "retiree")


    #create system prompt
    system_prompt = get_system_prompt(persona)

    #create messages
    messages = [SystemMessage(content=system_prompt)] + state["messages"]


    #invoke LLM with tools
    response = llm_with_tools.invoke(messages)

    #return the updated messages ( this bappends AI msg to history)
    return {"messages":[response]}


def should_continue(state:AgentState) :
    """
        Checks if the last message has tool calls. If the last message is the END token, we stop.
    """
    
    last_message = state["messages"][-1]

    if last_message.tool_calls:
            return "tools"
    
    return END


## Build the Graph

def create_retirement_graph(specific_tools=None):
    """
        Creates the Retirement Advisor Graph.
    """
    ## if upload file sends new tool to create_retirement_graph, it will be assigned as specific_tools other wise tools list is used
    specific_tools = specific_tools if specific_tools else tools

    llm_with_tools = llm.bind_tools(specific_tools)

    # creating the blueprint workflow
    workflow = StateGraph(AgentState)

    # This creates a new version of the function where 'llm' is already set
    rchestrator_func = partial(orchestrator_node, llm_with_tools=llm_with_tools)

    #Now start adding nodes, start with orchestrator
    workflow.add_node("orchestrator", rchestrator_func)


    # Add Tool Nodes as tools list
    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)

    ##set entry point
    workflow.set_entry_point("orchestrator")


    # Positional: 1st argument (source)
    # Positional: 2nd argument (path function)
    # Positional: 3rd argument (destinations)
    ##add edges
    workflow.add_conditional_edges(
        "orchestrator",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )

    ## from tools back to orchestrator
    workflow.add_edge("tools", "orchestrator")

    return workflow.compile()


# --- TEST IT ---
if __name__ == "__main__":
    app = create_retirement_graph()

    print("Retirement Advisor Graph Test:", app)

    print(app.get_graph().draw_mermaid())
    
    # Test Query 1: General Chat
    print("--- Test 1: Chat ---")
    res = app.invoke({"messages": [HumanMessage(content="Hi, I'm just starting to plan for retirement.")]})
    print(res["messages"][-1].content)

    # Test Query 2: Math Tool
    print("\n--- Test 2: Math ---")
    res = app.invoke({"messages": [HumanMessage(content="If I save $5000 a year for 20 years at 7%, how much will I have?")]})
    print(res["messages"][-1].content)