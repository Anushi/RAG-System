"""Streamlit UI for the Document RAG Assistant."""
import streamlit as st
from rag import RAGPipeline

st.set_page_config(page_title="Document RAG Assistant", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model and indexing documents...")
def get_pipeline():
    return RAGPipeline()


rag = get_pipeline()

st.title("📚 Document RAG Assistant")
st.caption("Ask a question. Answers are generated only from the retrieved passages.")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Passages to retrieve (top-k)", 1, 10, rag.top_k)
    st.write(f"**Embedding model:** `{rag.embed_model_name}`")
    st.write(f"**LLM:** `{rag.llm_model}`")
    st.write(f"**Chunks indexed:** {len(rag.chunks)}")

question = st.text_input(
    "Your question",
    placeholder="e.g. What are the main stages of a RAG pipeline?",
)

if question:
    with st.spinner("Retrieving and generating..."):
        result = rag.ask(question, top_k)

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Retrieved sources")
    for i, (chunk, score) in enumerate(result["sources"], 1):
        with st.expander(f"[{i}] {chunk.source} (chunk {chunk.chunk_id}) - similarity {score:.3f}"):
            st.write(chunk.text)
