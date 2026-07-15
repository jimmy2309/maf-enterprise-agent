import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from src.modules.verification.agent.state import AgentState
from src.modules.verification.agent.tools import AVAILABLE_TOOLS, check_criminal_record, lookup_company_policy
from src.db.vector_store import get_qdrant_client
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# Initialize the LLM (Groq)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# --- 1. SEMANTIC TOOL ROUTING (Handling 55+ Tools) ---
def semantic_tool_selector(user_query: str) -> list:
    """
    Demonstrates Semantic Routing.
    Instead of giving the LLM all 55+ tools (which crashes context limits),
    we mathematically find the best tool for the specific query.
    """
    # For a real 55+ tool system, you'd query Qdrant here.
    # We will simulate the semantic check logic:
    query = user_query.lower()
    selected_tools = []
    
    # If the user asks about rules, leaves, or policies -> Only give the Policy Tool (RAG)
    if any(word in query for word in ["policy", "rule", "leave", "handbook", "guideline"]):
        selected_tools.append(lookup_company_policy)
    
    # If the user asks about records, crime, or background -> Only give Criminal Tool
    if any(word in query for word in ["record", "crime", "criminal", "background", "check"]):
        selected_tools.append(check_criminal_record)
        
    # Fallback if semantic routing doesn't find an exact match
    if not selected_tools:
        selected_tools = AVAILABLE_TOOLS
        
    return selected_tools

# --- 2. LCEL IMPLEMENTATION ---
def call_model(state: AgentState):
    """The main reasoning node that uses LCEL"""
    messages = state["messages"]
    last_user_message = [m for m in messages if m.type == "human"][-1].content
    
    # Apply Semantic Tool Routing
    smart_tools = semantic_tool_selector(last_user_message)
    
    # This is LCEL in action! We bind the dynamically selected tools to the LLM.
    llm_with_tools = llm.bind_tools(smart_tools)
    
    # Invoke the LLM with the conversation history
    response = llm_with_tools.invoke(messages)
    
    # Return the new message to be appended to the state
    return {"messages": [response]}

# The ToolNode automatically handles calling the python functions when the LLM requests them
tool_node = ToolNode(AVAILABLE_TOOLS)

def should_continue(state: AgentState) -> str:
    """Conditional Edge: Decides if we need to run a tool or if we are done."""
    last_message = state["messages"][-1]
    
    # If the LLM decided to call a tool, route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    
    # Otherwise, it gave a final answer, so route to "END"
    return "END"
