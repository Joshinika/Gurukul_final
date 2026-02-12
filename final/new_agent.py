import os
import json
from typing import TypedDict, List, Dict
from neo4j import GraphDatabase
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# ======================================================
# MODELS
# ======================================================
agent_llm = ChatOllama(model="gemma-3-4b-it:latest", temperature=0)
answer_llm = ChatOllama(model="gemma-3-4b-it:latest", temperature=0)

# ======================================================
# VECTOR STORE
# ======================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "vectorstore", "chroma_amazon_reviews")

embeddings = OllamaEmbeddings(model="gte-large")

vector_db = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
    collection_name="amazon_reviews"
)

# ======================================================
# NEO4J
# ======================================================
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password")
)

# ======================================================
# STATE
# ======================================================
class AgentState(TypedDict):
    query: str
    docs: List[Dict]
    ranked_docs: List[Dict]
    answer: str
    validated: bool
    recommendation: str
    confidence: float
    iteration: int
    provenance: List[Dict]

# ======================================================
# CYPHER
# ======================================================
CYPHER_PROMPT = """
You are a Neo4j Cypher expert.
Return ONLY valid Cypher.
Limit 20.
"""

def generate_cypher(question: str) -> str:
    res = agent_llm.invoke([
        SystemMessage(content=CYPHER_PROMPT),
        HumanMessage(content=question)
    ])

    text = res.content.replace("```", "").strip()
    if not text.upper().startswith("MATCH"):
        return "MATCH (n) RETURN n LIMIT 5"

    return text

def run_cypher(query: str) -> List[Dict]:
    try:
        with driver.session() as session:
            result = session.run(query)
            return [
                {"text": str(dict(r)), "source": "graph", "score": 1.0}
                for r in result
            ]
    except Exception as e:
        print("Cypher failed:", e)
        return []

# ======================================================
# RETRIEVE
# ======================================================
def retrieve_node(state: AgentState) -> Dict:
    print("Running RETRIEVE")

    graph_docs = run_cypher(generate_cypher(state["query"]))

    vec = vector_db.similarity_search(state["query"], k=5)
    vec_docs = [
        {
            "text": d.page_content[:500],
            "source": "vector",
            "score": 0.7
        }
        for d in vec
    ]

    docs = graph_docs + vec_docs

    state.setdefault("provenance", []).extend(docs)

    return {
        "docs": docs,
        "iteration": state.get("iteration", 0)
    }

# ======================================================
# RANK
# ======================================================
def rank_node(state: AgentState) -> Dict:
    print("Running RANK")

    ranked = state["docs"][:5]

    return {
        "ranked_docs": ranked,
        "confidence": 0.8
    }

# ======================================================
# ANSWER
# ======================================================
ANSWER_PROMPT = """
Answer using ONLY provided context.
If insufficient evidence say: Insufficient information.
"""

def answer_node(state: AgentState) -> Dict:
    print("Running ANSWER")

    res = answer_llm.invoke([
        SystemMessage(content=ANSWER_PROMPT),
        HumanMessage(content=json.dumps({
            "query": state["query"],
            "context": state["ranked_docs"]
        }))
    ])

    return {"answer": res.content}

# ======================================================
# CRITIC
# ======================================================
def critic_node(state: AgentState) -> Dict:
    print("Running CRITIC")

    # simple validation: ensure answer not empty
    valid = bool(state.get("answer"))

    return {
        "validated": valid,
        "confidence": 0.9 if valid else 0.5
    }

# ======================================================
# RECOMMENDATION
# ======================================================
def recommendation_node(state: AgentState) -> Dict:
    print("Running RECOMMEND")

    return {
        "recommendation": "Would you like a comparison with another brand?"
    }

# ======================================================
# SUPERVISOR (SAFE VERSION)
# ======================================================
def supervisor_router(state: AgentState) -> str:
    iteration = state.get("iteration", 0)
    MAX_ITER = 2

    # Hard stop guard
    if iteration >= MAX_ITER:
        if not state.get("answer"):
            return "answer"
        if not state.get("recommendation"):
            return "recommend"
        return END

    if not state.get("docs"):
        return "retrieve"

    if not state.get("ranked_docs"):
        return "rank"

    if not state.get("answer"):
        return "answer"

    if not state.get("validated"):
        state["iteration"] = iteration + 1
        return "critic"

    if not state.get("recommendation"):
        return "recommend"

    return END

# ======================================================
# WORKFLOW
# ======================================================
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rank", rank_node)
workflow.add_node("answer", answer_node)
workflow.add_node("critic", critic_node)
workflow.add_node("recommend", recommendation_node)

workflow.set_entry_point("retrieve")

for node in ["retrieve", "rank", "answer", "critic"]:
    workflow.add_conditional_edges(
        node,
        supervisor_router,
        {
            "retrieve": "retrieve",
            "rank": "rank",
            "answer": "answer",
            "critic": "critic",
            "recommend": "recommend",
            END: END,
        }
    )

workflow.add_edge("recommend", END)

app = workflow.compile()
