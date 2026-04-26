"""
FastAPI + LangGraph Agent with Tools
LangSmith observability via LANGCHAIN_TRACING_V2 (auto-tracing)
"""

import os
import re
import traceback
from typing import Annotated, TypedDict, List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Security, Request, Query, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# ============================================================
# ENV
# ============================================================
load_dotenv()

GOOGLE_API_KEY    = os.getenv("GOOGLE_API_KEY", "")
API_KEY           = os.getenv("API_KEY", "")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY", "")

# LangSmith auto-tracing — no decorators needed
# Just set env vars and every LangChain/LangGraph call is traced automatically
if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"]   = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"]    = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGCHAIN_PROJECT", "agrigpt-backend-agent")
    print(f"[LangSmith] Auto-tracing enabled -> {os.environ['LANGCHAIN_PROJECT']}")
else:
    print("[LangSmith] No API key — tracing disabled")

# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(title="AgriGPT Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(key: str = Security(api_key_header)):
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")

# ============================================================
# GLOBAL STORAGE & HISTORY
# ============================================================
global_tool_results: List[Dict[str, Any]] = []
in_memory_history:   Dict[str, list]       = {}

def load_history(chat_id: str) -> list:
    return in_memory_history.get(chat_id, [])

def save_history(chat_id: str, messages: list):
    in_memory_history[chat_id] = messages

# ============================================================
# TOOLS
# ============================================================
def simulate_pests(crop_name: str, location: str = "general") -> str:
    """Simulates pest and disease activity for a given crop and location."""
    pest_data = {
        "rice":      "Possible Blast disease or Stem Borer activity detected. Treatment: Use Tricyclazole for blast and Chlorantraniliprole for stem borer.",
        "wheat":     "Risk of Rust disease. Maintain proper irrigation and use fungicides if yellow spots appear.",
        "tomato":    "Early Blight likely due to humidity. Increase spacing and apply copper-based fungicides.",
        "cotton":    "Pink Bollworm alert! Use pheromone traps and avoid late sowing.",
        "maize":     "Fall Armyworm detected. Apply Emamectin Benzoate or Spinetoram. Check leaves for egg masses.",
        "sugarcane": "Risk of Red Rot and Woolly Aphid infestation. Use resistant varieties and spray Dimethoate.",
        "soybean":   "Soybean Mosaic Virus risk. Control aphid vectors with Imidacloprid and remove infected plants.",
        "groundnut": "Leaf Spot and Tikka disease alert. Apply Mancozeb every 10-15 days during humid conditions.",
        "sunflower": "Downy Mildew risk detected. Use metalaxyl-treated seeds and avoid waterlogging.",
        "chilli":    "Thrips and Mite infestation likely. Spray Abamectin or Fipronil. Avoid water stress.",
        "onion":     "Purple Blotch and Thrips alert. Apply Mancozeb + Carbendazim and maintain field hygiene.",
        "potato":    "Late Blight high risk due to cool and moist conditions. Apply Cymoxanil + Mancozeb immediately.",
        "mustard":   "Aphid and Alternaria Blight warning. Spray Oxydemeton-Methyl and remove crop debris post-harvest.",
        "banana":    "Panama Wilt (Fusarium) risk. Use disease-free suckers and apply Trichoderma to soil.",
        "mango":     "Mango Hopper and Powdery Mildew alert. Spray Imidacloprid for hoppers and Wettable Sulfur for mildew.",
        "grapes":    "Downy and Powdery Mildew risk high. Apply Fosetyl-Al and Hexaconazole alternately.",
        "peas":      "Powdery Mildew and Pod Borer likely. Use Karathane for mildew and Indoxacarb for borers.",
        "lentils":   "Stemphylium Blight and Aphid attack. Spray Iprodione and use yellow sticky traps.",
        "cabbage":   "Diamondback Moth (DBM) alert. Use Spinosad or NSKE 5% spray. Practice crop rotation.",
        "brinjal":   "Shoot and Fruit Borer infestation. Apply Emamectin Benzoate and remove affected shoots promptly.",
        "cucumber":  "Downy Mildew and Red Pumpkin Beetle risk. Spray Chlorothalonil and use ash around plant base.",
        "chickpea":  "Helicoverpa Pod Borer high risk. Apply Indoxacarb or HaNPV. Monitor using pheromone traps.",
        "jowar":     "Shoot Fly and Aphid alert. Treat seeds with Imidacloprid and spray Dimethoate at early growth.",
        "bajra":     "Downy Mildew and Ergot disease risk. Use resistant hybrids and apply Metalaxyl seed treatment.",
        "turmeric":  "Rhizome Rot and Leaf Blotch detected. Treat rhizomes with Mancozeb before planting.",
        "ginger":    "Soft Rot (Pythium) risk high. Drench soil with Copper Oxychloride and ensure good drainage.",
        "coffee":    "White Stem Borer and Berry Borer alert. Use Chlorpyrifos and maintain shade tree management.",
        "tea":       "Blister Blight and Red Spider Mite risk. Apply Hexaconazole and Dicofol respectively.",
        "coconut":   "Rhinoceros Beetle and Bud Rot alert. Apply Carbaryl and remove decaying organic matter.",
        "papaya":    "Papaya Ringspot Virus via aphids. Remove infected plants and control aphids with Mineral Oil spray."
    }
    result = pest_data.get(
        crop_name.lower(),
        f"No specific pest data for {crop_name}. Monitor regularly for unusual leaf patterns or insects."
    )
    return f"Pest Simulation Results for {crop_name} in {location}: {result}"


def get_government_schemes(state: str = "India") -> str:
    """Retrieves information on agricultural government schemes and subsidies."""
    schemes = [
        "PM-KISAN: Financial support of Rs.6,000 per year to small and marginal farmers.",
        "PM Fasal Bima Yojana: Affordable crop insurance for farmers against natural calamities.",
        "Soil Health Card Scheme: Helps farmers understand soil nutrient status.",
        "Kisan Credit Card (KCC): Provides timely credit to farmers for cultivation needs.",
        "Agriculture Infrastructure Fund (AIF): Rs.1 lakh crore financing for post-harvest infrastructure.",
        "National Agricultural Market (eNAM): Online trading platform for better price discovery.",
        "Pradhan Mantri Krishi Sinchayee Yojana (PMKSY): Promotes micro-irrigation for water efficiency.",
        "Per Drop More Crop: Drip and sprinkler irrigation with subsidy up to 55% for small farmers.",
        "Paramparagat Krishi Vikas Yojana (PKVY): Promotes organic farming with Rs.50,000/hectare support.",
        "Rashtriya Krishi Vikas Yojana (RKVY): Holistic agriculture development with flexible funding.",
        "Digital Agriculture Mission: Promotes AI, IoT, and remote sensing for precision farming.",
        "National Food Security Mission (NFSM): Increases production of rice, wheat, and pulses.",
        "Interest Subvention Scheme: Crop loans up to Rs.3 lakh at 7% interest rate.",
        "Pradhan Mantri Kisan MaanDhan Yojana: Pension of Rs.3,000/month to farmers after age 60.",
        "Pradhan Mantri Matsya Sampada Yojana: Rs.20,000 crore scheme for fisheries sector.",
        "Rashtriya Gokul Mission: Develops indigenous bovine breeds for higher milk productivity.",
        "Rythu Bandhu (Telangana): Rs.10,000 per acre per year to farming land owners.",
        "Rythu Bima (Telangana): Free life insurance of Rs.5 lakh to farmers aged 18-59.",
        "Mission Kakatiya (Telangana): Restoration of tanks and minor irrigation sources.",
        "Telangana Micro Irrigation Project: 100% subsidy for drip and sprinkler irrigation."
    ]
    return f"Active Government Schemes for {state}: " + " | ".join(schemes)


def get_gemini_fallback(query: str) -> tuple[str, str]:
    """Call Gemini API directly when tools don't find answers."""
    try:
        llm_fb = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            google_api_key=GOOGLE_API_KEY,
        )
        response = llm_fb.invoke([
            SystemMessage(content="You are an expert agricultural assistant."),
            HumanMessage(content=query)
        ])
        answer = response.content if hasattr(response, "content") else str(response)
        return answer, "success"
    except Exception as e:
        return f"Unable to generate answer: {str(e)}", "error"


pest_simulation_tool = StructuredTool.from_function(
    func=simulate_pests,
    name="simulate_pests",
    description="Simulates pest and disease activity for a given crop and location."
)
government_schemes_tool = StructuredTool.from_function(
    func=get_government_schemes,
    name="government_schemes",
    description="Retrieves information on agricultural government schemes and subsidies."
)
TOOLS = [pest_simulation_tool, government_schemes_tool]

# ============================================================
# LANGGRAPH AGENT
# ============================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]

llm_agent      = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=GOOGLE_API_KEY)
llm_with_tools = llm_agent.bind_tools(TOOLS, tool_choice="auto")

def agent_node(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def should_continue(state: State):
    last = state["messages"][-1]
    return "tools" if (hasattr(last, "tool_calls") and last.tool_calls) else END

def tool_node(state: State):
    global global_tool_results
    last    = state["messages"][-1]
    outputs = []
    for call in last.tool_calls:
        name    = call["name"]
        args    = call.get("args", {})
        tool_id = call.get("id", "")
        tool    = next((t for t in TOOLS if t.name == name), None)
        result  = tool.invoke(args) if tool else f"Tool {name} not found"
        global_tool_results.append({"tool": name, "result": result})
        outputs.append(ToolMessage(content=str(result), tool_call_id=tool_id, name=name))
    return {"messages": outputs}

print("\nBUILDING AGENT...")
workflow  = StateGraph(State)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
app_agent = workflow.compile()
print("AGENT READY\n")

# ============================================================
# HELPERS
# ============================================================
def extract_final_answer(result: dict) -> str:
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            if isinstance(msg.content, str) and msg.content.strip():
                return msg.content
            elif isinstance(msg.content, list) and msg.content:
                block = msg.content[0]
                if isinstance(block, dict) and block.get("text", "").strip():
                    return block["text"]
    return "No response generated."

def clean_response_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    text = text.replace("\\n", "\n")
    if "Sources:" in text:
        text = text.split("Sources:")[0]
    return text.strip()

def has_meaningful_results(tool_results):
    for tr in tool_results:
        result = tr.get("result", "")
        if isinstance(result, str) and len(result.strip()) > 10:
            return True
    return False

# ============================================================
# MODELS
# ============================================================
class ChatRequest(BaseModel):
    chatId:       str
    phone_number: str
    message:      str

class ChatResponse(BaseModel):
    chatId:       str
    phone_number: str
    response:     str
    sources:      List[str] = []

# ============================================================
# ROUTES
# ============================================================
@app.get("/hi", tags=["Health"])
async def hi():
    return {"message": "Hi from AgriGPT!"}

@app.get("/webhook")
async def verify_webhook(
    hub_mode:         str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge:    str = Query(None, alias="hub.challenge"),
):
    if hub_mode == "subscribe" and hub_verify_token == "test_verify_token_123":
        return PlainTextResponse(content=hub_challenge, status_code=200)
    raise HTTPException(status_code=403, detail="Webhook verification failed.")

@app.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()
    print(f"[Webhook] Payload: {payload}")
    return {"status": "ok"}

# ============================================================
# MAIN CHAT ENDPOINT
# ============================================================
@app.post("/test/chat", response_model=ChatResponse)
def test_chat(request: ChatRequest, _ = Depends(verify_api_key)):
    """
    Chat endpoint — LangSmith traces automatically via LANGCHAIN_TRACING_V2=true.
    Every LangGraph run, LLM call, and tool call is captured without decorators.
    """
    global global_tool_results
    global_tool_results = []

    print(f"\n[chat] chatId={request.chatId} | msg={request.message[:60]}")

    try:
        history = load_history(request.chatId)
        history = [m for m in history if not isinstance(m, SystemMessage)]
        history = [SystemMessage(content="""You are AgriGPT, an agricultural assistant.
Use simulate_pests for pest/disease questions.
Use government_schemes for subsidy/scheme questions.
Write in PLAIN TEXT only.""")] + history
        history.append(HumanMessage(content=request.message))

        # This entire invoke is auto-traced by LangSmith
        result    = app_agent.invoke({"messages": history})
        final_ans = extract_final_answer(result)
        save_history(request.chatId, result["messages"])

        if has_meaningful_results(global_tool_results):
            sources = list({t["tool"] for t in global_tool_results}) or ["Knowledge Base"]
        else:
            gemini_ans, status = get_gemini_fallback(request.message)
            final_ans = f"Based on general agricultural knowledge:\n\n{gemini_ans}" if status == "success" else f"Could not retrieve: {gemini_ans}"
            sources   = ["Gemini API"] if status == "success" else ["Error"]

        cleaned = clean_response_text(final_ans)
        print(f"[chat] sources={sources}")

        return ChatResponse(
            chatId=request.chatId,
            phone_number=request.phone_number,
            response=cleaned,
            sources=sources,
        )

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, _ = Depends(verify_api_key)):
    """Production chat endpoint."""
    return test_chat(request)

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8030)
