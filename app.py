"""
FastAPI + LangGraph Agent with Tools (FIXED VERSION)
"""

import os
import json
import traceback
from typing import Annotated, TypedDict, List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from langsmith import Client

# ============================================================
# ENV
# ============================================================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
API_KEY = os.getenv("API_KEY", "")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY", "")

# ============================================================
# APP (MUST BE BEFORE DECORATORS)
# ============================================================
app = FastAPI(title="AgriGPT Fixed Backend")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(key: str = Security(api_key_header)):
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

ls_client = Client() if langsmith_api_key else None

# ============================================================
# GLOBAL TOOL STORAGE
# ============================================================
global_tool_results = []

# ============================================================
# TOOLS (FIXED)
# ============================================================

def simulate_pests(crop_name: str, location: str = "general") -> str:
    """Simulates pest and disease activity for crops."""
    pest_data = {
        "rice": "Blast disease / Stem borer detected. Use Tricyclazole or Chlorantraniliprole.",
        "wheat": "Rust disease risk. Use fungicides and proper irrigation.",
        "tomato": "Early blight risk. Use copper fungicides.",
        "cotton": "Pink bollworm attack. Use pheromone traps.",
        "maize": "Fall armyworm detected. Use Emamectin Benzoate.",
        "sugarcane": "Red rot risk. Use resistant varieties.",
        "chilli": "Thrips infestation. Use Fipronil spray."
    }

    return pest_data.get(
        crop_name.lower(),
        f"No specific pest data for {crop_name}. Monitor regularly."
    )


def get_government_schemes(state: str = "India") -> str:
    """Returns agricultural schemes."""
    schemes = [
        "PM-KISAN: ₹6000/year income support",
        "PM Fasal Bima Yojana: Crop insurance",
        "Kisan Credit Card: Low interest loans",
        "Soil Health Card: Soil testing program",
        "eNAM: Online crop market platform"
    ]
    return f"Schemes for {state}: " + " | ".join(schemes)


# REQUIRED FIX → description MUST exist
pest_tool = StructuredTool.from_function(
    func=simulate_pests,
    name="simulate_pests",
    description="Detect crop pests and diseases and suggest treatments."
)

schemes_tool = StructuredTool.from_function(
    func=get_government_schemes,
    name="government_schemes",
    description="Get agriculture government schemes in India."
)

TOOLS = [pest_tool, schemes_tool]

# ============================================================
# STATE
# ============================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ============================================================
# LLM
# ============================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
)

llm_with_tools = llm.bind_tools(TOOLS)

# ============================================================
# AGENT
# ============================================================
def agent_node(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def should_continue(state: State):
    last = state["messages"][-1]
    return "tools" if hasattr(last, "tool_calls") and last.tool_calls else END


def tool_node(state: State):
    global global_tool_results

    last = state["messages"][-1]
    outputs = []

    for call in last.tool_calls:
        name = call["name"]
        args = call.get("args", {})
        tool_id = call.get("id")

        tool = next(t for t in TOOLS if t.name == name)
        result = tool.invoke(args)

        global_tool_results.append({
            "tool": name,
            "result": result
        })

        outputs.append(
            ToolMessage(content=str(result), tool_call_id=tool_id)
        )

    return {"messages": outputs}


# build graph
graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

app_agent = graph.compile()

# ============================================================
# MODELS
# ============================================================
class ChatRequest(BaseModel):
    chatId: str
    phone_number: str
    message: str

class ChatResponse(BaseModel):
    chatId: str
    phone_number: str
    response: str
    sources: List[str] = []

# ============================================================
# CHAT ENDPOINT
# ============================================================
@app.post("/test/chat", response_model=ChatResponse)
def chat(request: ChatRequest, _=Depends(verify_api_key)):
    global global_tool_results
    global_tool_results = []

    system = SystemMessage(content="You are AgriGPT. Use tools when needed.")

    messages = [
        system,
        HumanMessage(content=request.message)
    ]

    result = app_agent.invoke({"messages": messages})

    final = ""
    for m in reversed(result["messages"]):
        if isinstance(m, AIMessage):
            final = m.content
            break

    sources = [t["tool"] for t in global_tool_results] or ["general knowledge"]

    return ChatResponse(
        chatId=request.chatId,
        phone_number=request.phone_number,
        response=final,
        sources=sources
    )

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8030)