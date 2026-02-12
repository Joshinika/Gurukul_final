import os
import json
import re
from typing import TypedDict, List
from neo4j import GraphDatabase
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# ======================================================
# MODELS
# ======================================================
agent_llm = ChatOllama(model="llama-3.2-3b-it:latest", temperature=0)
answer_llm = ChatOllama(model="llama-3.2-3b-it:latest", temperature=0)

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
    docs: List[str]
    ranked_docs: List[str]
    answer: str
    validated: bool
    recommendation: str

# ======================================================
# CYPHER PROMPT
# ======================================================
CYPHER_PROMPT = """
You are a Neo4j Cypher expert.

Database Schema:
(Brand)-[:MAKES]->(Product)
(Product)-[:HAS_REVIEW]->(Review)

Brand {name}
Product {name, price}
Review {rating, votes, text}

Important Rules:
• rating exists ONLY in Review node (r.rating)
• Always traverse Product -> Review when using rating
• Example for average rating:

MATCH (b:Brand)-[:MAKES]->(p:Product)-[:HAS_REVIEW]->(r:Review)
RETURN b.name, AVG(r.rating)

• Cypher DOES NOT use GROUP BY
• Return ONLY Cypher query text
• No explanations
• Limit 20
"""

def generate_cypher(question):
    res = agent_llm.invoke([
        SystemMessage(content=CYPHER_PROMPT),
        HumanMessage(content=question)
    ])

    text = res.content.replace("```", "").strip()
    lines = text.split("\n")

    clean = []
    started = False
    for l in lines:
        if l.upper().startswith(("MATCH","WITH","RETURN","CALL","UNWIND")):
            started = True
        if started:
            clean.append(l)

    cypher = "\n".join(clean).strip()
    if not cypher:
        cypher = "MATCH (n) RETURN n LIMIT 5"

    return cypher

def run_cypher(query):
    query = query.replace("GROUP BY", "").replace("group by", "")

    try:
        with driver.session() as session:
            result = session.run(query)
            rows = []

            for r in result:
                d = dict(r)
                brand = d.get("b.name") or d.get("brand")
                rating = d.get("AVG(r.rating)") or d.get("avg_rating")

                if brand and rating:
                    rows.append(f"Brand {brand} has average rating {rating}")
                else:
                    rows.append(str(d))

            return rows

    except Exception as e:
        print("Cypher failed → fallback to vector retrieval:", e)
        return []


# ======================================================
# RETRIEVE
# ======================================================
def retrieve_node(state):
    cypher = generate_cypher(state["query"])
    graph_docs = run_cypher(cypher)

    vec = vector_db.similarity_search(state["query"], k=5)
    vec_docs = [
        f"Brand:{d.metadata['brand']} Product:{d.metadata['product']} Review:{d.page_content}"
        for d in vec
    ]

    docs = graph_docs + vec_docs[:3]
    return { "docs": docs}


# ======================================================
# RANK
# ======================================================
RANK_PROMPT = """
You are the RankerAgent.

Input:
• user_query
• retrieved_context

Rules:
• Always keep structured graph results (numeric values, aggregations)
• Select only context useful for answering the query
• Remove unrelated reviews or brands
• Do NOT generate new information"

Return strictly JSON:
{
 "top_context": [...]
}
"""

def rank_node(state):

    response = agent_llm.invoke([
        SystemMessage(content=RANK_PROMPT),
        HumanMessage(content=json.dumps({
            "user_query": state["query"],
            "retrieved_context": state["docs"]
        }))
    ])

    try:
        ranked = json.loads(response.content)["top_context"]
    except:
        ranked = state["docs"][:5]

    return {"ranked_docs": ranked}



# ======================================================
# ANSWER
# ======================================================
ANSWER_PROMPT = """
You are the AnswerAgent.

Rules:
• Answer the user's question using ONLY the provided context.
• If strong factual or numeric evidence exists, present it clearly and directly.
• If only partial evidence exists, provide the closest supported answer without speculation.
• Never invent or hallucinate missing numbers, facts, or details.
• Summarize in a natural, conversational tone that feels human and approachable.
• If multiple options or brands exist, highlight the top few with concise comparisons.
• Always end naturally in a chatbot style — never leave the answer blank or abrupt.
• If no evidence exists, politely say "Insufficient information" in plain human language.
• Conclude with a helpful conversational follow-up to keep the dialogue engaging.
"""


def answer_node(state):

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
CRITIC_PROMPT = """
Check if answer is grounded in context.
Return JSON {"valid":true/false}
"""

def critic_node(state):
    res = agent_llm.invoke([
        SystemMessage(content=CRITIC_PROMPT),
        HumanMessage(content=json.dumps({
            "answer": state["answer"],
            "context": state["ranked_docs"]
        }))
    ])
    try:
        valid = json.loads(res.content)["valid"]
    except:
        valid = True
    return {"validated": valid}


# ======================================================
# RECOMMENDATION
# ======================================================
RECOMMEND_PROMPT = """
You are the RecommendationAgent.

Goal:
Suggest ONE natural conversational follow-up question
that helps the user explore the result further.

Rules:
• Use the answer provided
• Suggest a helpful next step (reviews, products, comparisons, pricing etc.)
• Keep it short and conversational
• Do not repeat the answer
Return only the question text
"""

def recommendation_node(state):

    res = agent_llm.invoke([
        SystemMessage(content=RECOMMEND_PROMPT),
        HumanMessage(content=json.dumps({
            "user_query": state["query"],
            "answer": state["answer"]
        }))
    ])

    return {"recommendation": res.content}


# ======================================================
# SUPERVISOR ROUTER
# ======================================================
SUPERVISOR_PROMPT = """
You are the SupervisorAgent controlling an autonomous AI workflow.

Available agents:
retrieve → gather graph + vector evidence
rank → filter useful evidence
answer → generate grounded answer
critic → verify answer groundedness
recommend → suggest follow-up question
end → finish workflow

Decide the BEST next agent based on:
• user query
• current state fields
• whether evidence is sufficient
• whether answer already exists
• whether validation is completed

Return JSON:
{"next":"agent_name"}
"""
def supervisor_router(state):

    # ---------- Mandatory progress guards ----------
    if not state.get("docs"):
        return "retrieve"

    if not state.get("ranked_docs"):
        return "rank"

    if not state.get("answer"):
        return "answer"

    if not state.get("validated"):
        return "critic"

    if not state.get("recommendation"):
        return "recommend"

    # ---------- Dynamic reasoning after completion ----------
    res = agent_llm.invoke([
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=json.dumps(state))
    ])

    try:
        return json.loads(res.content)["next"]
    except:
        return "end"



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

workflow.add_conditional_edges(
    "retrieve",
    supervisor_router,
    {"retrieve":"retrieve","rank":"rank","answer":"answer","critic":"critic","recommend":"recommend","end":END}
)

workflow.add_conditional_edges(
    "rank",
    supervisor_router,
    {"retrieve":"retrieve","rank":"rank","answer":"answer","critic":"critic","recommend":"recommend","end":END}
)

workflow.add_conditional_edges(
    "answer",
    supervisor_router,
    {"retrieve":"retrieve","rank":"rank","answer":"answer","critic":"critic","recommend":"recommend","end":END}
)

workflow.add_conditional_edges(
    "critic",
    supervisor_router,
    {"retrieve":"retrieve","rank":"rank","answer":"answer","critic":"critic","recommend":"recommend","end":END}
)

workflow.add_edge("recommend", END)

app = workflow.compile()
