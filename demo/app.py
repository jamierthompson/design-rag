"""
Streamlit demo UI for DesignRAG.

A simple web interface for asking questions and seeing cited answers.
Runs the RAG pipeline directly (no API server needed).

Usage:
    uv run streamlit run demo/app.py
"""

import streamlit as st

from design_rag.metadata import DocumentType, TopicArea
from design_rag.retrieval.qa import ask

# Page configuration
st.set_page_config(
    page_title="DesignRAG",
    page_icon="📐",
    layout="wide",
)

st.title("DesignRAG")
st.markdown(
    "Ask questions about interior design standards, pricing, "
    "meeting procedures, and more — answers are grounded in "
    "source documents with citations."
)

# Sidebar — filters and settings
with st.sidebar:
    st.header("Settings")

    # Topic area filter (optional)
    topic_options = ["All topics"] + [t.value for t in TopicArea]
    selected_topic = st.selectbox("Filter by topic area", topic_options)

    # Document type filter (optional)
    doc_type_options = ["All types"] + [d.value for d in DocumentType]
    selected_doc_type = st.selectbox("Filter by document type", doc_type_options)

    # Number of chunks to retrieve
    n_results = st.slider("Chunks to retrieve", min_value=1, max_value=20, value=5)

    st.divider()
    st.markdown("**Example questions:**")
    st.markdown("- How do I calculate the FIE?")
    st.markdown("- What is the revision policy?")
    st.markdown("- What products clean marble?")
    st.markdown("- How many bids for large projects?")

# Build filters dict from sidebar selections
filters: dict[str, str] = {}
if selected_topic != "All topics":
    filters["topic_area"] = selected_topic
if selected_doc_type != "All types":
    filters["document_type"] = selected_doc_type

# Main input
question = st.text_input(
    "Ask a question",
    placeholder="e.g., How do I calculate the Furnishings Investment Estimate?",
)

if question:
    with st.spinner("Searching documents and generating answer..."):
        result = ask(
            question=question,
            n_results=n_results,
            filters=filters if filters else None,
        )

    # Display the answer
    st.subheader("Answer")
    st.markdown(result["answer"])

    # Display metadata
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Tokens used", f"{result['tokens_used']:,}")
    with col2:
        st.metric("Sources found", len(result["sources"]))

    # Display sources
    if result["sources"]:
        st.subheader("Sources")
        for i, source in enumerate(result["sources"], start=1):
            with st.expander(
                f"Source {i}: {source['file']} "
                f"(page {source['page']}, "
                f"relevance: {source['relevance_score']:.2f})"
            ):
                st.markdown(source["content"])
