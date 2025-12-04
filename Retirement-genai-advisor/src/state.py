import operator
from typing import Annotated, List, TypedDict, Union
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


# # Defines how updates to the state are merged (we append messages, not overwrite)
# def add_messages(left: list, right: list):
#     return left + right

## using add_messages from langgraph.graph instead of operator.add and add_message function implementation

class AgentState(TypedDict):
    # The conversation history (User, AI, Tool messages)
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Track which persona is active ('retiree', 'financial_planner', 'family_member'.)
    user_persona: str