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

API_KEY           = os.getenv("API_KEY", "")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY", "")

def get_google_api_key() -> str:
    """Read GOOGLE_API_KEY lazily at call time so Render env vars are available."""
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("GOOGLE_API_KEY is not set. Add it to your Render environment variables.")
    return key

# LangSmith auto-tracing — no decorators needed
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
# KNOWLEDGE BASES  (single source of truth used by both the
#                   /knowledge-base/* endpoints AND the tools)
# ============================================================

PESTS_KNOWLEDGE_BASE: Dict[str, Dict[str, str]] = {
    "rice":      {"pests": "Stem Borer, Brown Plant Hopper", "diseases": "Blast, Sheath Blight", "treatment": "Use Tricyclazole for blast and Chlorantraniliprole for stem borer."},
    "wheat":     {"pests": "Aphids, Hessian Fly", "diseases": "Rust (Yellow/Brown/Black), Loose Smut", "treatment": "Maintain proper irrigation and use fungicides if yellow spots appear."},
    "tomato":    {"pests": "Whitefly, Fruit Borer", "diseases": "Early Blight, Late Blight, Leaf Curl Virus", "treatment": "Increase spacing and apply copper-based fungicides."},
    "cotton":    {"pests": "Pink Bollworm, Bollweevil, Whitefly", "diseases": "Cotton Leaf Curl Virus, Fusarium Wilt", "treatment": "Use pheromone traps and avoid late sowing."},
    "maize":     {"pests": "Fall Armyworm, Stem Borer", "diseases": "Downy Mildew, Maize Streak Virus", "treatment": "Apply Emamectin Benzoate or Spinetoram. Check leaves for egg masses."},
    "sugarcane": {"pests": "Woolly Aphid, Top Shoot Borer", "diseases": "Red Rot, Smut, Ratoon Stunting", "treatment": "Use resistant varieties and spray Dimethoate."},
    "soybean":   {"pests": "Aphids, Pod Borer", "diseases": "Soybean Mosaic Virus, Rust", "treatment": "Control aphid vectors with Imidacloprid and remove infected plants."},
    "groundnut": {"pests": "Thrips, Leaf Miner", "diseases": "Leaf Spot, Tikka Disease, Stem Rot", "treatment": "Apply Mancozeb every 10–15 days during humid conditions."},
    "sunflower": {"pests": "Capitulum Borer, Aphids", "diseases": "Downy Mildew, Alternaria Blight", "treatment": "Use metalaxyl-treated seeds and avoid waterlogging."},
    "chilli":    {"pests": "Thrips, Mites, Aphids", "diseases": "Anthracnose, Leaf Curl, Powdery Mildew", "treatment": "Spray Abamectin or Fipronil. Avoid water stress."},
    "onion":     {"pests": "Thrips, Bulb Mite", "diseases": "Purple Blotch, Stemphylium Blight", "treatment": "Apply Mancozeb + Carbendazim and maintain field hygiene."},
    "potato":    {"pests": "Aphids, Tuber Moth", "diseases": "Late Blight, Early Blight, Black Scurf", "treatment": "Apply Cymoxanil + Mancozeb immediately in cool moist conditions."},
    "mustard":   {"pests": "Aphids, Painted Bug", "diseases": "Alternaria Blight, White Rust, Downy Mildew", "treatment": "Spray Oxydemeton-Methyl and remove crop debris post-harvest."},
    "banana":    {"pests": "Banana Weevil, Nematodes", "diseases": "Panama Wilt (Fusarium), Sigatoka", "treatment": "Use disease-free suckers and apply Trichoderma to soil."},
    "mango":     {"pests": "Mango Hopper, Fruit Fly", "diseases": "Powdery Mildew, Anthracnose, Die-back", "treatment": "Spray Imidacloprid for hoppers and Wettable Sulfur for mildew."},
    "grapes":    {"pests": "Mealybug, Thrips", "diseases": "Downy Mildew, Powdery Mildew, Anthracnose", "treatment": "Apply Fosetyl-Al and Hexaconazole alternately."},
    "chickpea":  {"pests": "Helicoverpa Pod Borer, Cutworm", "diseases": "Ascochyta Blight, Fusarium Wilt", "treatment": "Apply Indoxacarb or HaNPV. Monitor using pheromone traps."},
    "coconut":   {"pests": "Rhinoceros Beetle, Eriophyid Mite", "diseases": "Bud Rot, Root Wilt", "treatment": "Apply Carbaryl and remove decaying organic matter."},
    "papaya":    {"pests": "Aphids, Mites", "diseases": "Papaya Ringspot Virus, Powdery Mildew", "treatment": "Remove infected plants and control aphids with Mineral Oil spray."},
}

SCHEMES_KNOWLEDGE_BASE: Dict[str, Dict[str, str]] = {
    "PM-KISAN": {
        "full_name": "Pradhan Mantri Kisan Samman Nidhi",
        "benefit": "Financial support of Rs.6,000 per year in three equal instalments.",
        "eligibility": "Small and marginal farmers with cultivable land.",
        "category": "income_support",
    },
    "PMFBY": {
        "full_name": "Pradhan Mantri Fasal Bima Yojana",
        "benefit": "Affordable crop insurance against natural calamities, pests and diseases.",
        "eligibility": "All farmers including sharecroppers and tenant farmers.",
        "category": "insurance",
    },
    "Soil Health Card": {
        "full_name": "Soil Health Card Scheme",
        "benefit": "Free soil testing and customized fertilizer recommendations every 2 years.",
        "eligibility": "All farmers.",
        "category": "advisory",
    },
    "KCC": {
        "full_name": "Kisan Credit Card",
        "benefit": "Short-term credit up to Rs.3 lakh at 7% interest (4% with prompt repayment).",
        "eligibility": "All farmers, fishermen, and animal husbandry farmers.",
        "category": "credit",
    },
    "AIF": {
        "full_name": "Agriculture Infrastructure Fund",
        "benefit": "Rs.1 lakh crore financing facility for post-harvest management infrastructure.",
        "eligibility": "FPOs, PACS, agri-entrepreneurs, start-ups.",
        "category": "infrastructure",
    },
    "eNAM": {
        "full_name": "National Agricultural Market",
        "benefit": "Online transparent trading platform across 1,000+ mandis for better price discovery.",
        "eligibility": "All farmers with produce registered at linked APMC mandis.",
        "category": "market_access",
    },
    "PMKSY": {
        "full_name": "Pradhan Mantri Krishi Sinchayee Yojana",
        "benefit": "Drip and sprinkler irrigation subsidy up to 55% for small/marginal farmers.",
        "eligibility": "All categories of farmers.",
        "category": "irrigation",
    },
    "PKVY": {
        "full_name": "Paramparagat Krishi Vikas Yojana",
        "benefit": "Rs.50,000 per hectare over 3 years to promote organic farming clusters.",
        "eligibility": "Groups of farmers (minimum 50 per cluster) for organic conversion.",
        "category": "organic_farming",
    },
    "RKVY": {
        "full_name": "Rashtriya Krishi Vikas Yojana",
        "benefit": "Flexible block grants to states for holistic agriculture development.",
        "eligibility": "State governments; benefits flow to farmers through state schemes.",
        "category": "development",
    },
    "PM Kisan MaanDhan": {
        "full_name": "Pradhan Mantri Kisan MaanDhan Yojana",
        "benefit": "Pension of Rs.3,000 per month after age 60.",
        "eligibility": "Small and marginal farmers aged 18–40 years.",
        "category": "pension",
    },
    "Rythu Bandhu": {
        "full_name": "Rythu Bandhu Scheme (Telangana)",
        "benefit": "Rs.10,000 per acre per year (Rs.5,000 each for Rabi and Kharif seasons).",
        "eligibility": "Landowning farmers in Telangana.",
        "category": "income_support",
    },
    "Rythu Bima": {
        "full_name": "Rythu Bima Scheme (Telangana)",
        "benefit": "Free life insurance cover of Rs.5 lakh to farmer families.",
        "eligibility": "Farmers aged 18–59 in Telangana.",
        "category": "insurance",
    },
    "Mission Kakatiya": {
        "full_name": "Mission Kakatiya (Telangana)",
        "benefit": "Restoration of tanks and minor irrigation sources to improve water availability.",
        "eligibility": "All farmers in Telangana benefiting from restored tanks.",
        "category": "irrigation",
    },
    "TM-IP": {
        "full_name": "Telangana Micro Irrigation Project",
        "benefit": "100% subsidy for drip and sprinkler irrigation systems.",
        "eligibility": "All farmers in Telangana.",
        "category": "irrigation",
    },
}

# ============================================================
# KNOWLEDGE BASE REQUEST / RESPONSE MODELS
# ============================================================

class PestQueryRequest(BaseModel):
    crop_name: str
    location:  str = "general"

class PestQueryResponse(BaseModel):
    crop_name: str
    location:  str
    pests:     str
    diseases:  str
    treatment: str
    source:    str = "Pests Knowledge Base"

class SchemeQueryRequest(BaseModel):
    state:    str    = "India"
    category: str    = ""   # optional filter e.g. "irrigation", "insurance"

class SchemeDetail(BaseModel):
    scheme_id:   str
    full_name:   str
    benefit:     str
    eligibility: str
    category:    str

class SchemeQueryResponse(BaseModel):
    state:   str
    count:   int
    schemes: List[SchemeDetail]
    source:  str = "Schemes Knowledge Base"

# ============================================================
# KNOWLEDGE BASE ENDPOINTS  (Microservices per architecture)
# ============================================================

@app.get("/knowledge-base/pests", tags=["Knowledge Base"])
def list_all_pests():
    """Returns all crops available in the Pests Knowledge Base."""
    return {
        "available_crops": sorted(PESTS_KNOWLEDGE_BASE.keys()),
        "total": len(PESTS_KNOWLEDGE_BASE),
        "source": "Pests Knowledge Base",
    }

@app.post("/knowledge-base/pests/query", response_model=PestQueryResponse, tags=["Knowledge Base"])
def query_pests_knowledge_base(request: PestQueryRequest):
    """
    Microservice endpoint — Pests Knowledge Base.
    Returns pest, disease, and treatment data for a given crop.
    Called internally by Tool 2 (simulate_pests).
    """
    crop_key = request.crop_name.strip().lower()
    data = PESTS_KNOWLEDGE_BASE.get(crop_key)
    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"No pest data found for crop '{request.crop_name}'. "
                   f"Available crops: {sorted(PESTS_KNOWLEDGE_BASE.keys())}",
        )
    return PestQueryResponse(
        crop_name=request.crop_name,
        location=request.location,
        pests=data["pests"],
        diseases=data["diseases"],
        treatment=data["treatment"],
    )

@app.get("/knowledge-base/schemes", tags=["Knowledge Base"])
def list_all_schemes():
    """Returns all scheme IDs available in the Schemes Knowledge Base."""
    return {
        "available_schemes": sorted(SCHEMES_KNOWLEDGE_BASE.keys()),
        "categories": sorted({v["category"] for v in SCHEMES_KNOWLEDGE_BASE.values()}),
        "total": len(SCHEMES_KNOWLEDGE_BASE),
        "source": "Schemes Knowledge Base",
    }

@app.post("/knowledge-base/schemes/query", response_model=SchemeQueryResponse, tags=["Knowledge Base"])
def query_schemes_knowledge_base(request: SchemeQueryRequest):
    """
    Microservice endpoint — Schemes Knowledge Base.
    Returns government schemes filtered by state and optional category.
    Called internally by Tool 1 (government_schemes).
    """
    results = []
    for scheme_id, details in SCHEMES_KNOWLEDGE_BASE.items():
        # Simple state filter: Telangana-specific schemes are flagged in full_name
        if request.state.lower() not in ("india", "") and \
           "telangana" in details["full_name"].lower() and \
           "telangana" not in request.state.lower():
            continue
        if request.category and details["category"].lower() != request.category.lower():
            continue
        results.append(SchemeDetail(
            scheme_id=scheme_id,
            full_name=details["full_name"],
            benefit=details["benefit"],
            eligibility=details["eligibility"],
            category=details["category"],
        ))

    return SchemeQueryResponse(
        state=request.state,
        count=len(results),
        schemes=results,
    )

# ============================================================
# TOOLS  (now backed by the knowledge-base endpoints above)
# ============================================================

def simulate_pests(crop_name: str, location: str = "general") -> str:
    """Simulates pest and disease activity for a given crop and location."""
    crop_key = crop_name.strip().lower()
    data = PESTS_KNOWLEDGE_BASE.get(crop_key)
    if not data:
        return (
            f"No specific pest data for '{crop_name}' in the knowledge base. "
            "Monitor regularly for unusual leaf patterns or insects."
        )
    return (
        f"Pest Simulation Results for {crop_name} in {location}: "
        f"Common Pests: {data['pests']}. "
        f"Diseases: {data['diseases']}. "
        f"Treatment: {data['treatment']}"
    )


def get_government_schemes(state: str = "India") -> str:
    """Retrieves information on agricultural government schemes and subsidies."""
    lines = []
    for scheme_id, details in SCHEMES_KNOWLEDGE_BASE.items():
        # Skip Telangana-specific schemes when state is not Telangana
        if "telangana" in details["full_name"].lower() and \
           "telangana" not in state.lower() and state.lower() != "india":
            continue
        lines.append(f"{details['full_name']}: {details['benefit']}")

    return f"Active Government Schemes for {state}: " + " | ".join(lines)


def get_gemini_fallback(query: str) -> tuple[str, str]:
    """Call Gemini API directly when tools don't find answers."""
    try:
        llm_fb = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            google_api_key=get_google_api_key(),
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

def get_llm_with_tools():
    """Lazily initialize LLM so GOOGLE_API_KEY is read at request time."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=get_google_api_key(),
    )
    return llm.bind_tools(TOOLS, tool_choice="auto")

def agent_node(state: State):
    return {"messages": [get_llm_with_tools().invoke(state["messages"])]}

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
