import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agents.agents import app

st.set_page_config(
    page_title="Hybrid Graph + Vector RAG",
    layout="wide",
    page_icon="📊"
)

# Sidebar
with st.sidebar:
    st.markdown("## 📊 AI Assistant")
    st.markdown("### Navigation")
    st.markdown("""
    • Hybrid Graph RAG  
    • Agentic AI Pipeline  
    • Vector + Neo4j Intelligence
    """)
    st.markdown("---")
    st.markdown("### System Info")
    st.success("Vector DB: Active")
    st.success("Neo4j Graph: Connected")
    st.info("LLM: Ollama Local")

# Header
st.markdown("""
<h1 style='text-align:center;margin-top:10px'>
📱 Hybrid Graph + Vector RAG Assistant
</h1>
<p style='text-align:center;color:gray'>
Enterprise Agentic AI Recommendation Platform
</p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Query panel
col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.markdown("<h4 style='text-align:center'>Ask your question</h4>", unsafe_allow_html=True)

    query = st.text_input("", placeholder="Example: Which brand has highest average rating?")
    run_btn = st.button("Run AI Agents", use_container_width=True)

# Run pipeline
if run_btn and query:

    with st.spinner("Running AI Agents..."):
        result = app.invoke({
            "query": query,
            "docs": [],
            "ranked_docs": [],
            "answer": "",
            "validated": False,
            "recommendation": ""
        })

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## Results")

    res1, res2 = st.columns([3,1])

    with res1:
        st.markdown(
        f"""
        <div style='background:#f5f7fb;padding:20px;border-radius:12px'>
        <h4>Query</h4>
        <p>{query}</p>
        <hr>
        <h4>Answer</h4>
        <p>{result.get('answer','No answer generated')}</p>
        </div>
        """,
        unsafe_allow_html=True
        )

    with res2:
        st.markdown(
        f"""
        <div style='background:#eaf7ef;padding:20px;border-radius:12px'>
        <h4>Recommendation</h4>
        <p>{result.get('recommendation','No recommendation')}</p>
        </div>
        """,
        unsafe_allow_html=True
        )
