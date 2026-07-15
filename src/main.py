import os
import sys

# Ensure 'src' can be imported regardless of where the script is run from
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langfuse.callback import CallbackHandler
from langchain_core.messages import HumanMessage
from src.modules.verification.agent.graph import enterprise_agent


app = FastAPI(title="MAF Enterprise Agent API")

# Allow the frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For POC only. In production, use specific origins like ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default_user_1"  # Used by memory to remember this specific user's chat

# --- VISUAL TRACING (The n8n experience) ---
# This handler automatically captures the entire LangGraph flow (Thoughts, Tool calls, Results)
# and sends it to your Langfuse dashboard so you can visually see the process step-by-step.
langfuse_handler = CallbackHandler()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # Create the config with the thread_id for memory, and attach the Langfuse tracer
    config = {
        "configurable": {"thread_id": request.thread_id},
        "callbacks": [langfuse_handler]
    }
    
    # Send the user's message into the LangGraph state machine
    inputs = {"messages": [HumanMessage(content=request.message)]}
    
    # Run the graph and stream/invoke the final result
    result = enterprise_agent.invoke(inputs, config=config)
    
    # Extract the final AI message from the state
    final_message = result["messages"][-1].content
    
    return {"reply": final_message}

if __name__ == "__main__":
    import uvicorn
    # Run the API on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
