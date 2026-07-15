from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.modules.verification.agent.state import AgentState
from src.modules.verification.agent.nodes import call_model, tool_node, should_continue

def create_agent_graph():
    # 1. Initialize the Graph with our State
    workflow = StateGraph(AgentState)

    # 2. Add our Nodes (Workers)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    # 3. Add Edges (The flow)
    workflow.add_edge("__start__", "agent")
    
    # Conditional Routing: After the agent thinks, does it need a tool or is it done?
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "END": END
        }
    )
    
    # After a tool runs, it ALWAYS goes back to the agent to read the observation
    workflow.add_edge("tools", "agent")

    # --- 3. MEMORY (CHECKPOINTER) IMPLEMENTATION ---
    # MemorySaver keeps track of chat history per thread_id. 
    # For production with PostgreSQL, you would use AsyncPostgresSaver here.
    checkpointer = MemorySaver()

    # Compile the graph with the checkpointer attached
    app = workflow.compile(checkpointer=checkpointer)
    
    return app

# Expose the compiled graph
enterprise_agent = create_agent_graph()
