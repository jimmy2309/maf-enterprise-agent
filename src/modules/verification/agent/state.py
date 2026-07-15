import operator
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # 'messages' will hold the entire conversation history.
    # The `operator.add` reducer means when we return new messages, they are APPENDED to the list, not overwritten.
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # We can add custom state variables here later if needed (e.g., user_id, current_workflow_step)
