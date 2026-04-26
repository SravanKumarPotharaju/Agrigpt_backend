"""
FastAPI + LangGraph Agent with Multi-MCP Tool Discovery
WhatsApp Business API (Meta Cloud API) Webhook Handler

FIXED VERSION: Uses Mohan's proven approach to capture and extract sources correctly.

Connects to MULTIPLE MCP servers simultaneously (e.g. Alumnx + Vignan)
and merges all their tools into one agent dynamically at startup.

CRITICAL: Agent ALWAYS calls tools first before answering.
System prompt forces tool usage with mandatory rules.

New Chat flow:
  - Frontend generates a new UUID on "New Chat" click and sends it as chat_id.
  - Backend finds no history for that chat_id → agent starts fresh.
  - MongoDB creates the document automatically on first save.
  - Same chat_id on subsequent messages → history is loaded and agent remembers.

Auto Deploy enabled using deploy.yml file
"""

import os
import httpx
import asyncio
import json
from datetime import datetime, timezone
from typing import Annotated, TypedDict, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request, Query, BackgroundTasks, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, create_model
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection

# ============================================================
# Environment
# ============================================================
load_dotenv()

langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"]   = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"]    = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"]    = "agrigpt-backend-agent"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MCP_TIMEOUT    = float(os.getenv("MCP_TIMEOUT", "30"))

# ── API Key Auth ─────────────────────────────────────────────
API_KEY        = os.getenv("API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(key: str = Security(api_key_header)):
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")

# ── Multi-MCP Configuration ──────────────────────────────────────────────────
# COMMENTED OUT MCP CONFIGURATION
# MCP_SERVERS: List[Dict[str, str]] = [
#     {
#         "name":    "Alumnx",
#         "url":     os.getenv("ALUMNX_MCP_URL", "").strip(),
#         "api_key": os.getenv("ALUMNX_MCP_API_KEY", "").strip(),
#     },
#     {
#         "name":    "Vignan",
#         "url":     os.getenv("VIGNAN_MCP_URL", "").strip(),
#         "api_key": os.getenv("VIGNAN_MCP_API_KEY", "").strip(),
#     },
# ]
MCP_SERVERS = []

MONGODB_URI        = os.getenv("MONGODB_URI")
MONGODB_DB         = os.getenv("MONGODB_DB", "agrigpt")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "chats")

MAX_MESSAGES = 20

# ============================================================
# GLOBAL STORAGE FOR RAW TOOL RESULTS
# This is the KEY FIX from Mohan's approach
# Stores raw dict results BEFORE any stringification
# ============================================================
global_tool_results: List[Dict[str, Any]] = []

# ============================================================
# MongoDB Setup (COMMENTED OUT)
# ============================================================
# mongo_client   = MongoClient(MONGODB_URI)
# db             = mongo_client[MONGODB_DB]
# chat_sessions: Collection = db[MONGODB_COLLECTION]

# chat_sessions.create_index([("chat_id",      ASCENDING)], unique=True)
# chat_sessions.create_index([("phone_number", ASCENDING)])
# chat_sessions.create_index([("updated_at",   ASCENDING)])

# print(f"Connected to MongoDB: {MONGODB_DB}.{MONGODB_COLLECTION}")

# SIMPLE IN-MEMORY MEMORY FOR HISTORY
in_memory_history = {}

# ============================================================
# MongoDB Memory Helpers
# ============================================================

def load_history(chat_id: str) -> list:
    """Load stored messages from in-memory history."""
    return in_memory_history.get(chat_id, [])

def save_history(chat_id: str, messages: list, phone_number: str | None = None):
    """Persist updated conversation history in-memory."""
    in_memory_history[chat_id] = messages


# ============================================================
# NEW SIMPLE TOOLS
# ============================================================

def simulate_pests(crop_name: str, location: str = "general") -> str:
    """
    Simulates pest and disease activity for a given crop and location.
    Provides identification and control measures.
    """
    pest_data = {
        "rice": "Possible Blast disease or Stem Borer activity detected. Treatment: Use Tricyclazole for blast and Chlorantraniliprole for stem borer.",
        "wheat": "Risk of Rust disease. Maintain proper irrigation and use fungicides if yellow spots appear.",
        "tomato": "Early Blight likely due to humidity. Increase spacing and apply copper-based fungicides.",
        "cotton": "Pink Bollworm alert! Use pheromone traps and avoid late sowing.",
        "maize": "Fall Armyworm detected. Apply Emamectin Benzoate or Spinetoram. Check leaves for egg masses.",
        "sugarcane": "Risk of Red Rot and Woolly Aphid infestation. Use resistant varieties and spray Dimethoate.",
        "soybean": "Soybean Mosaic Virus risk. Control aphid vectors with Imidacloprid and remove infected plants.",
        "groundnut": "Leaf Spot and Tikka disease alert. Apply Mancozeb every 10-15 days during humid conditions.",
        "sunflower": "Downy Mildew risk detected. Use metalaxyl-treated seeds and avoid waterlogging.",
        "chilli": "Thrips and Mite infestation likely. Spray Abamectin or Fipronil. Avoid water stress.",
        "onion": "Purple Blotch and Thrips alert. Apply Mancozeb + Carbendazim and maintain field hygiene.",
        "potato": "Late Blight high risk due to cool and moist conditions. Apply Cymoxanil + Mancozeb immediately.",
        "mustard": "Aphid and Alternaria Blight warning. Spray Oxydemeton-Methyl and remove crop debris post-harvest.",
        "banana": "Panama Wilt (Fusarium) risk. Use disease-free suckers and apply Trichoderma to soil.",
        "mango": "Mango Hopper and Powdery Mildew alert. Spray Imidacloprid for hoppers and Wettable Sulfur for mildew.",
        "grapes": "Downy and Powdery Mildew risk high. Apply Fosetyl-Al and Hexaconazole alternately.",
        "peas": "Powdery Mildew and Pod Borer likely. Use Karathane for mildew and Indoxacarb for borers.",
        "lentils": "Stemphylium Blight and Aphid attack. Spray Iprodione and use yellow sticky traps.",
        "cabbage": "Diamondback Moth (DBM) alert. Use Spinosad or NSKE 5% spray. Practice crop rotation.",
        "brinjal": "Shoot and Fruit Borer infestation. Apply Emamectin Benzoate and remove affected shoots promptly.",
        "cucumber": "Downy Mildew and Red Pumpkin Beetle risk. Spray Chlorothalonil and use ash around plant base.",
        "chickpea": "Helicoverpa Pod Borer high risk. Apply Indoxacarb or HaNPV. Monitor using pheromone traps.",
        "jowar": "Shoot Fly and Aphid alert. Treat seeds with Imidacloprid and spray Dimethoate at early growth.",
        "bajra": "Downy Mildew and Ergot disease risk. Use resistant hybrids and apply Metalaxyl seed treatment.",
        "turmeric": "Rhizome Rot and Leaf Blotch detected. Treat rhizomes with Mancozeb before planting.",
        "ginger": "Soft Rot (Pythium) risk high. Drench soil with Copper Oxychloride and ensure good drainage.",
        "coffee": "White Stem Borer and Berry Borer alert. Use Chlorpyrifos and maintain shade tree management.",
        "tea": "Blister Blight and Red Spider Mite risk. Apply Hexaconazole and Dicofol respectively.",
        "coconut": "Rhinoceros Beetle and Bud Rot alert. Apply Carbaryl and remove decaying organic matter.",
        "papaya": "Papaya Ringspot Virus via aphids. Remove infected plants and control aphids with Mineral Oil spray."
    }
    result = pest_data.get(crop_name.lower(), f"No specific pest simulation data for {crop_name}. Advice: Monitor regularly for unusual leaf patterns or insects.")
    return f"Pest Simulation Results for {crop_name} in {location}: {result}"

def get_government_schemes(state: str = "India") -> str:
    """
    Retrieves information on agricultural government schemes and subsidies.
    """
    schemes = [
        "PM-KISAN: Financial support of Rs.6,000 per year to small and marginal farmers.",
        "PM Fasal Bima Yojana: Affordable crop insurance for farmers against natural calamities.",
        "Soil Health Card Scheme: Helps farmers understand soil nutrient status and recommended dosage of fertilizers.",
        "Kisan Credit Card (KCC): Provides timely credit to farmers for their cultivation and other needs.",
        "Agriculture Infrastructure Fund (AIF): Rs.1 lakh crore financing facility for post-harvest infrastructure and community farming assets.",
        "PM Kisan Samman Nidhi: Direct income support transferred directly to bank accounts of eligible farmer families.",
        "National Agricultural Market (eNAM): Online trading platform linking APMCs for better price discovery for farmers.",
        "Pradhan Mantri Krishi Sinchayee Yojana (PMKSY): Ensures Har Khet Ko Pani and promotes micro-irrigation for water use efficiency.",
        "Per Drop More Crop: Promotes drip and sprinkler irrigation systems with subsidy up to 55% for small farmers.",
        "Atal Bhujal Yojana: Groundwater management scheme focused on water-stressed areas across 7 states.",
        "Watershed Development Component: Develops rainfed areas through integrated management of natural resources.",
        "Paramparagat Krishi Vikas Yojana (PKVY): Promotes organic farming through cluster-based approach with Rs.50,000/hectare support.",
        "National Mission on Oilseeds and Oil Palm (NMOOP): Increases oilseed production with financial assistance and technology.",
        "Sub-Mission on Agricultural Mechanization (SMAM): Promotes farm mechanization by providing machinery at subsidized rates.",
        "Rashtriya Krishi Vikas Yojana (RKVY): Holistic development of agriculture with state-specific plans and flexible funding.",
        "Digital Agriculture Mission: Promotes AI, IoT, and remote sensing in agriculture for precision farming.",
        "National e-Governance Plan in Agriculture (NeGP-A): Delivers information and services to farmers via ICT tools.",
        "National Food Security Mission (NFSM): Increases production of rice, wheat, pulses, and coarse cereals through area expansion.",
        "Mission for Integrated Development of Horticulture (MIDH): Promotes holistic growth of horticulture including fruits, vegetables, and spices.",
        "National Mission for Sustainable Agriculture (NMSA): Enhances agricultural productivity in rainfed areas with focus on soil health.",
        "Integrated Scheme on Agricultural Cooperation (ISAC): Strengthens cooperative movement in agriculture and sugar sectors.",
        "Cotton Development Program: Provides quality seeds, pest management, and technology support to cotton growers.",
        "Jute-ICARE: Improves jute cultivation practices and provides certified seeds to jute farmers.",
        "Interest Subvention Scheme: Provides short-term crop loans up to Rs.3 lakh at 7% interest rate (4% for prompt repayment).",
        "Pradhan Mantri Kisan MaanDhan Yojana (PM-KMY): Pension scheme providing Rs.3,000/month to small and marginal farmers after age 60.",
        "Agri Clinics and Agri Business Centres (ACABC): Supports agricultural graduates to set up agri-ventures with subsidized loans.",
        "NABARD Farmer Finance: Provides refinance support to banks for agricultural and rural development lending.",
        "Pradhan Mantri Kisan Sampada Yojana: Develops modern infrastructure for food processing and reduces post-harvest losses.",
        "Operation Greens: Stabilizes supply of Tomato, Onion, and Potato (TOP) and extends to all fruits and vegetables.",
        "Horticulture Mission for North East and Himalayan States (HMNEH): Supports horticulture development in NE and Himalayan regions with focus on organic farming.",
        "Rastriya Goukul Mission: Aims at dairy development and genetic upgradation of indigenous cattle breeds.",
        "Livestock Health and Disease Control (LH&DC): Provides financial assistance for disease control programs and livestock health management.",
        "Gramin Bhandaran Yojana: Creates scientific storage capacity in rural areas with subsidy for warehouse construction.",
        "Market Intervention Scheme (MIS): Provides price support for perishable commodities when market prices fall sharply.",
        "Price Support Scheme (PSS): Procures oilseeds, pulses, and cotton at MSP when prices fall below minimum support price.",
        "Pradhan Mantri Matsya Sampada Yojana (PMMSY): Rs.20,000 crore scheme to bring blue revolution in fisheries sector.",
        "Animal Husbandry Infrastructure Development Fund (AHIDF): Rs.15,000 crore fund for dairy, meat, and animal feed processing.",
        "Rashtriya Gokul Mission: Conserves and develops indigenous bovine breeds for higher milk productivity.",
        "National Beekeeping and Honey Mission (NBHM): Promotes scientific beekeeping for additional income and pollination support.",
        "Rythu Bandhu (Telangana): Investment support of Rs.10,000 per acre per year to all farming land owners in Telangana.",
        "Rythu Bima (Telangana): Free life insurance of Rs.5 lakh to all farmers in Telangana aged 18-59 years.",
        "Mission Kakatiya (Telangana): Restoration of tanks and minor irrigation sources across Telangana for water conservation.",
        "Telangana Micro Irrigation Project: Promotes drip and sprinkler irrigation with 100% subsidy for small and marginal farmers."
    ]
    return f"Active Government Schemes for {state}: " + " | ".join(schemes)

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

# ============================================================
# MCP Client (COMMENTED OUT)
# ============================================================
# class MCPClient:
#     """REST client matching MCP servers' custom endpoint format."""
# ... (rest of class commented out)


# ============================================================
# LangGraph State
# ============================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# Agent Builder
# ============================================================
def build_agent():
    # USE ONLY OUR SIMPLE TOOLS
    all_tools = [pest_simulation_tool, government_schemes_tool]

    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=GOOGLE_API_KEY,
    )
    llm_with_tools = llm.bind_tools(all_tools, tool_choice="auto")

    # LangGraph nodes
    def agent_node(state: State):
        return {
            "messages": [llm_with_tools.invoke(state["messages"])],
        }

    def should_continue(state: State):
        last = state["messages"][-1]
        return "tools" if (hasattr(last, "tool_calls") and last.tool_calls) else END

    # ========== KEY FIX: Tool execution node that captures RAW results ==========
    def tool_execution_node(state: State):
        """Execute tools and capture RAW results to global_tool_results."""
        global global_tool_results

        messages = state["messages"]
        last_message = messages[-1]

        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return {"messages": []}

        tool_results_messages = []

        # Execute each tool call
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call.get("args", {})
            tool_id = tool_call.get("id", "")

            try:
                # Find and execute the tool
                tool_to_run = None
                for tool in all_tools:
                    if tool.name == tool_name:
                        tool_to_run = tool
                        break

                if tool_to_run:
                    # CAPTURE RAW RESULT BEFORE STRINGIFICATION
                    result = tool_to_run.invoke(tool_input)

                    print(f"[tool_execution] {tool_name} returned result")
                    print(f"[tool_execution] Result type: {type(result)}")

                    # Debug: Log the actual structure for source extraction
                    if isinstance(result, list) and len(result) > 0:
                        print(f"[tool_execution] Result is list, first item keys: {result[0].keys() if isinstance(result[0], dict) else 'N/A'}")
                        print(f"[tool_execution] First item sample: {str(result[0])[:200]}")
                    elif isinstance(result, dict):
                        print(f"[tool_execution] Result is dict, keys: {result.keys()}")

                    # STORE RAW DICT IN GLOBAL (THIS IS THE KEY FIX)
                    tool_result_item = {
                        'tool': tool_name,
                        'result': result,
                        'full_result': result
                    }
                    global_tool_results.append(tool_result_item)
                    print(f"[tool_execution] Stored raw result for {tool_name}")

                    # Create ToolMessage with stringified result (for LangGraph flow)
                    result_str = json.dumps(result) if isinstance(result, dict) else str(result)
                    tool_message = ToolMessage(
                        content=result_str,
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                    tool_results_messages.append(tool_message)

            except Exception as e:
                print(f"[tool_execution] Error executing {tool_name}: {e}")
                import traceback
                traceback.print_exc()

                error_result = {
                    "status": "error",
                    "message": str(e),
                    "sources": []
                }

                # Store error too
                global_tool_results.append({
                    'tool': tool_name,
                    'result': error_result,
                    'full_result': error_result
                })

                tool_message = ToolMessage(
                    content=str(error_result),
                    tool_call_id=tool_id,
                    name=tool_name
                )
                tool_results_messages.append(tool_message)

        return {"messages": tool_results_messages}

    workflow = StateGraph(State)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_execution_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    return workflow.compile()


# ============================================================
# Startup
# ============================================================
print("\nBUILDING AGENT AT STARTUP...")
app_agent = build_agent()
print("AGENT BUILD COMPLETE\n")


# ============================================================
# Gemini Fallback Handler
# ============================================================
def get_gemini_fallback(query: str) -> tuple[str, str]:
    """Call Gemini API directly when tools don't find answers."""
    print(f"[gemini_fallback] Calling Gemini for query: {query[:60]}")
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            google_api_key=GOOGLE_API_KEY,
        )

        response = llm.invoke([
            SystemMessage(content="You are an expert agricultural assistant. Provide clear, detailed answers about agriculture, crops, pests, and farming practices."),
            HumanMessage(content=query)
        ])

        answer = response.content if hasattr(response, 'content') else str(response)
        print(f"[gemini_fallback] Got answer from Gemini ({len(answer)} chars)")
        return answer, "success"

    except Exception as e:
        print(f"[gemini_fallback] Error calling Gemini: {e}")
        return f"Unable to generate answer: {str(e)}", "error"


# ============================================================
# Source Extraction (Using Mohan's proven approach)
# ============================================================
def extract_sources_from_tool_results(tool_results: List[Dict[str, Any]]) -> List[str]:
    """
    Extract source filenames directly from RAW tool results.

    Handles multiple result formats:
    - Dict with 'sources' field
    - Dict with 'results' field
    - Direct list results (from tools like VignanUniversity)

    For list results, we report them as sources if they contain meaningful data.
    """
    sources = set()

    if not tool_results:
        print("[extract_sources] No tool results provided")
        return []

    print(f"[extract_sources] Processing {len(tool_results)} tool results")

    for tool_result in tool_results:
        if not isinstance(tool_result, dict):
            continue

        tool_name = tool_result.get("tool", "unknown")
        result_data = tool_result.get("full_result") or tool_result.get("result")

        if not result_data:
            print(f"[extract_sources] {tool_name}: No result data")
            continue

        # Handle list results directly (e.g., VignanUniversity returns list)
        if isinstance(result_data, list):
            if len(result_data) > 0:
                print(f"[extract_sources] {tool_name}: Got list with {len(result_data)} items")
                for item in result_data:
                    if isinstance(item, dict):
                        source = (item.get("source") or
                                item.get("document") or
                                item.get("filename") or
                                item.get("pdf"))
                        if source:
                            source_str = str(source).strip()
                            if source_str:
                                sources.add(source_str)
                                print(f"    -> {source_str}")
                        elif item.get("metadata"):
                            metadata = item["metadata"]
                            if isinstance(metadata, dict):
                                source = (metadata.get("source") or
                                        metadata.get("document") or
                                        metadata.get("filename"))
                                if source:
                                    sources.add(str(source).strip())
                                    print(f"    -> {source}")
                if len(sources) == 0:
                    sources.add(tool_name)
                    print(f"    -> {tool_name} (no specific source found)")
            continue

        # Handle stringified JSON
        if isinstance(result_data, str):
            print(f"[extract_sources] {tool_name}: Result is string, parsing JSON...")
            try:
                result_data = json.loads(result_data)
            except:
                print(f"[extract_sources] {tool_name}: Could not parse, skipping")
                continue

        if not isinstance(result_data, dict):
            continue

        print(f"[extract_sources] {tool_name}:")

        # Extract from 'sources' field
        if "sources" in result_data:
            src_list = result_data["sources"]
            if isinstance(src_list, list):
                print(f"  Found 'sources' with {len(src_list)} items")
                for src in src_list:
                    if isinstance(src, dict) and "filename" in src:
                        filename = src["filename"]
                        if filename and isinstance(filename, str):
                            filename = filename.strip()
                            if filename:
                                sources.add(filename)
                                print(f"    -> {filename}")
                    elif isinstance(src, str) and src.strip():
                        sources.add(src.strip())
                        print(f"    -> {src.strip()}")

        # Extract from 'results' field
        if "results" in result_data:
            res_list = result_data["results"]
            if isinstance(res_list, list):
                print(f"  Found 'results' with {len(res_list)} items")
                for res in res_list:
                    if isinstance(res, dict) and "source" in res:
                        src = res["source"]
                        if isinstance(src, str) and src.strip():
                            sources.add(src.strip())
                            print(f"    -> {src}")

    final_sources = sorted(list(sources))
    print(f"[extract_sources] FINAL: {final_sources}")
    return final_sources


def clean_response_text(text: str) -> str:
    """Clean response text by removing markdown formatting."""
    if not text:
        return ""

    import re

    cleaned = re.sub(r'```[\s\S]*?```', '', text)
    cleaned = re.sub(r'`([^`]+)`', r'\1', cleaned)
    cleaned = re.sub(r'^#{1,6}\s+', '', cleaned)
    cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
    cleaned = re.sub(r'\*([^*]+)\*', r'\1', cleaned)
    cleaned = re.sub(r'__([^_]+)__', r'\1', cleaned)
    cleaned = re.sub(r'_([^_]+)_', r'\1', cleaned)
    cleaned = cleaned.replace("\\n", "\n")

    if "Sources:" in cleaned:
        cleaned = cleaned.split("Sources:")[0]

    cleaned = cleaned.strip()
    return cleaned


def extract_final_answer(result: dict) -> str:
    """Extract final text answer from agent result."""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            if isinstance(msg.content, str) and msg.content.strip():
                return msg.content
            elif isinstance(msg.content, list) and msg.content:
                block = msg.content[0]
                if isinstance(block, dict) and block.get("text", "").strip():
                    return block["text"]
                elif str(block).strip():
                    return str(block)
    return "No response generated."


def has_meaningful_tool_results(tool_results: List[Dict[str, Any]]) -> bool:
    """Check if tool results contain meaningful information."""
    if not tool_results:
        print("[has_meaningful_tool_results] No tool results")
        return False

    for tool_result in tool_results:
        if not isinstance(tool_result, dict):
            continue

        result_data = tool_result.get("full_result") or tool_result.get("result")

        # Recognize non-empty strings as meaningful (for our simple tools)
        if isinstance(result_data, str):
            if len(result_data.strip()) > 10:
                print(f"[has_meaningful_tool_results] Found meaningful string result")
                return True
            continue

        if not isinstance(result_data, dict):
            continue

        # Check for error status
        if result_data.get("status") == "error":
            continue

        # Check for 'sources'
        if result_data.get("sources") and isinstance(result_data["sources"], list):
            if len(result_data["sources"]) > 0:
                print(f"[has_meaningful_tool_results] Found sources")
                return True

        # Check for 'information'
        if result_data.get("information"):
            info_text = str(result_data["information"])
            if len(info_text) > 50:
                print(f"[has_meaningful_tool_results] Found information")
                return True

        # Check for 'results'
        if result_data.get("results") and isinstance(result_data["results"], list):
            if len(result_data["results"]) > 0:
                print(f"[has_meaningful_tool_results] Found results")
                return True

    print(f"[has_meaningful_tool_results] No meaningful results")
    return False


# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(title="AgriGPT Agent")

# ── CORS Middleware ───────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/webhook")
async def verify_webhook(
    hub_mode:         str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge:    str = Query(None, alias="hub.challenge"),
):
    LOCAL_VERIFY_TOKEN = "test_verify_token_123"
    if hub_mode == "subscribe" and hub_verify_token == LOCAL_VERIFY_TOKEN:
        print("Webhook verified successfully.")
        return PlainTextResponse(content=hub_challenge, status_code=200)
    raise HTTPException(status_code=403, detail="Webhook verification failed.")


@app.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    """Receives WhatsApp events. Returns 200 immediately, processes in background."""
    payload = await request.json()
    print(f"[Webhook] Incoming payload: {payload}")
    try:
        entry    = payload.get("entry", [{}])[0]
        changes  = entry.get("changes", [{}])[0]
        value    = changes.get("value", {})
        messages = value.get("messages", [])

        if not messages:
            return {"status": "ok"}

        message  = messages[0]
        msg_type = message.get("type")
        if msg_type != "text":
            return {"status": "ok"}

        phone_number = message.get("from")
        user_message = message["text"].get("body", "").strip()
        if not phone_number or not user_message:
            return {"status": "ok"}

        print(f"[Webhook] Message from {phone_number}: {user_message}")

    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"[Webhook] Parse error: {exc}")

    return {"status": "ok"}


@app.get("/hi", summary="Say Hi", tags=["Health"])
async def hi():
    """Returns a greeting."""
    return {"message": "Hi from AgriGPT!"}


# ============================================================
# Chat Endpoint Models
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
# MAIN CHAT ENDPOINT
# ============================================================
@app.post("/test/chat", response_model=ChatResponse)
def test_chat(request: ChatRequest, _ = Depends(verify_api_key)):
    """
    Chat endpoint with TOOL-FIRST then GEMINI-FALLBACK strategy.

    FIX: Uses global_tool_results to capture RAW results for proper source extraction.
    Protected by X-API-Key header.
    """
    global global_tool_results

    print(f"\n[/test/chat] ========== START REQUEST ==========")
    print(f"[/test/chat] chatId={request.chatId} | phone={request.phone_number}")
    print(f"[/test/chat] message={request.message[:60]}")

    try:
        # Clear previous tool results for this request
        global_tool_results.clear()

        # Load history
        history = load_history(request.chatId)
        print(f"[/test/chat] Loaded {len(history)} messages from history.")

        system_prompt = SystemMessage(content="""You are AgriGPT, a simple agricultural assistant.

YOUR MISSION: Provide accurate, helpful answers using your tools.

TOOL USAGE:
1. Use 'simulate_pests' for any questions about pests, diseases, or crop health.
2. Use 'government_schemes' for questions about subsidies or agricultural programs.

RESPONSE FORMATTING:
- Write in PLAIN TEXT only - NO markdown
- Be concise and helpful""")

        history = [msg for msg in history if not isinstance(msg, SystemMessage)]
        history = [system_prompt] + history
        history.append(HumanMessage(content=request.message))

        # ========== STEP 1: Invoke agent ==========
        print("\n[STEP 1] Invoking agent...")
        result = app_agent.invoke({"messages": history})
        print(f"[STEP 1] Agent returned {len(result['messages'])} messages")

        final_answer = extract_final_answer(result)

        # Save history
        save_history(request.chatId, result["messages"], phone_number=request.phone_number)

        # ========== STEP 2: Check for meaningful results ==========
        print("\n[STEP 2] Checking tool results...")
        sources = []

        has_meaningful = has_meaningful_tool_results(global_tool_results)
        print(f"[STEP 2] Has meaningful results: {has_meaningful}")

        # ========== STEP 3: Extract sources or use Gemini fallback ==========
        print("\n[STEP 3] Source strategy...")

        if has_meaningful:
            print("[STEP 3] Tools found results - extracting sources")
            sources = extract_sources_from_tool_results(global_tool_results)
            if not sources:
                sources = ["Knowledge Base"]
        else:
            print("[STEP 3] Tools found no results - using Gemini fallback")
            gemini_answer, gemini_status = get_gemini_fallback(request.message)

            if gemini_status == "success":
                final_answer = f"I couldn't find specific information in the knowledge base. Based on general agricultural knowledge:\n\n{gemini_answer}"
                sources = ["Gemini API"]
            else:
                final_answer = f"I couldn't retrieve information: {gemini_answer}"
                sources = ["Error - Unable to retrieve"]

        # ========== STEP 4: Clean response ==========
        cleaned_response = clean_response_text(final_answer)

        print(f"[STEP 4] FINAL SOURCES: {sources}")
        print(f"[/test/chat] ========== END REQUEST ==========\n")

        return ChatResponse(
            chatId=request.chatId,
            phone_number=request.phone_number,
            response=cleaned_response,
            sources=sources,
        )
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"[/test/chat] ========== REQUEST FAILED ==========\n")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, _ = Depends(verify_api_key)):
    """Production chat endpoint. Protected by X-API-Key header."""
    return test_chat(request)


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8030)