import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

from rag_chain import TravelRAGChain


BASE_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = BASE_DIR / "app_analytics.json"


st.set_page_config(
    page_title="Global Travel Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .stChatInputContainer { padding-bottom: 20px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_analytics():
    if LOG_FILE.exists():
        try:
            with LOG_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_analytics(data):
    with LOG_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def log_interaction(interaction):
    data = load_analytics()
    data.append(interaction)
    save_analytics(data)


def update_feedback(interaction_id, feedback):
    data = load_analytics()

    for entry in data:
        if entry.get("interaction_id") == interaction_id:
            entry["feedback"] = feedback
            break

    save_analytics(data)


@st.cache_resource
def load_rag_chain():
    return TravelRAGChain()


try:
    rag = load_rag_chain()
except Exception as e:
    st.error(f"Unable to start the travel assistant: {e}")
    st.stop()


tab_chat, tab_dashboard = st.tabs(
    ["💬 Copilot Chat", "📊 System & Monitoring Dashboard"]
)


with tab_chat:
    col1, col2 = st.columns([4, 1])

    with col1:
        st.title("✈️ Global Travel Assistant")
        st.markdown(
            "Your AI companion featuring Hybrid Search, Re-ranking, and Query Rewriting."
        )

    with col2:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")

    with st.sidebar:
        st.header("🌍 Explore & Discover")
        st.write("Choose a quick prompt or type your own question.")
        st.markdown("---")
        st.markdown("### Quick Prompts")

        if st.button("🍜 Best food spots in Tokyo", use_container_width=True):
            st.session_state.quick_prompt = (
                "What are the best food spots and dining etiquette in Tokyo?"
            )

        if st.button("🚇 Transit guide for Paris", use_container_width=True):
            st.session_state.quick_prompt = (
                "How do I navigate the transit system efficiently in Paris?"
            )

        if st.button("💎 Hidden gems in Rome", use_container_width=True):
            st.session_state.quick_prompt = (
                "What are some lesser-known hidden gems to visit in Rome?"
            )

        st.markdown("---")

        with st.expander("🛠️ Advanced Architecture", expanded=True):
            st.markdown("🟢 **LLM:** Qwen2.5 (0.5B)")
            st.markdown("🟢 **Search:** Hybrid (BM25 + Vector) + Re-ranking")
            n_results = st.slider(
                "Retrieved Chunks (K)",
                min_value=1,
                max_value=5,
                value=3,
            )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("interaction_id"):
                feedback_key = f"fb_{message['interaction_id']}"
                user_feedback = st.feedback("thumbs", key=feedback_key)

                if user_feedback is not None and message.get("feedback") != user_feedback:
                    message["feedback"] = user_feedback
                    update_feedback(message["interaction_id"], user_feedback)
                    st.toast("Feedback recorded!", icon="⭐")

    active_prompt = getattr(st.session_state, "quick_prompt", None)

    if active_prompt:
        del st.session_state.quick_prompt
        user_input = active_prompt
    else:
        user_input = st.chat_input("Ask about food, transit, culture...")

    if user_input:
        interaction_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat(timespec="seconds")

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner(
                "Processing through hybrid search, re-ranking & generation..."
            ):
                try:
                    result = rag.query(user_input, n_results=n_results)

                    st.markdown(result["answer"])

                    interaction = {
                        "interaction_id": interaction_id,
                        "timestamp": timestamp,
                        "query": user_input,
                        "rewritten_query": result["rewritten_query"],
                        "feedback": None,
                        "retrieval_strategy": result["retrieval_strategy"],
                        "retrieved_documents": result["retrieved_documents"],
                        "embedding_latency_ms": result["embedding_latency_ms"],
                        "bm25_latency_ms": result["bm25_latency_ms"],
                        "rerank_latency_ms": result["rerank_latency_ms"],
                        "generation_latency_ms": result["generation_latency_ms"],
                        "total_latency_ms": result["total_latency_ms"],
                        "sources": result["sources"],
                        "error": None,
                    }

                    log_interaction(interaction)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "timestamp": timestamp,
                            "interaction_id": interaction_id,
                            "query_ref": user_input,
                            "feedback": None,
                        }
                    )

                    with st.expander("🔎 RAG Trace"):
                        st.write("**Original query:**", result["original_query"])
                        st.write("**Rewritten query:**", result["rewritten_query"])
                        st.write("**Retrieval strategy:**", result["retrieval_strategy"])
                        st.write(
                            "**Retrieved documents:**",
                            result["retrieved_documents"],
                        )
                        st.write("**Sources:**")

                        for source in result["sources"]:
                            st.write(
                                f"- {source['city']} — "
                                f"{source['category']} — "
                                f"{source['title']}"
                            )

                        st.write(
                            f"**Total latency:** {result['total_latency_ms']:.0f} ms"
                        )

                    st.rerun()

                except Exception as e:
                    interaction = {
                        "interaction_id": interaction_id,
                        "timestamp": timestamp,
                        "query": user_input,
                        "rewritten_query": None,
                        "feedback": None,
                        "retrieval_strategy": None,
                        "retrieved_documents": 0,
                        "embedding_latency_ms": 0,
                        "bm25_latency_ms": 0,
                        "rerank_latency_ms": 0,
                        "generation_latency_ms": 0,
                        "total_latency_ms": 0,
                        "sources": [],
                        "error": str(e),
                    }

                    log_interaction(interaction)

                    st.error(
                        f"I encountered a temporary issue. Please try again.\n\n{e}"
                    )


with tab_dashboard:
    st.header("📊 System Monitoring & Telemetry Dashboard")
    st.write(
        "Real application telemetry covering traffic, latency, retrieval, errors, and feedback."
    )

    logs = load_analytics()

    total_queries = len(logs)
    successful_logs = [x for x in logs if not x.get("error")]
    failed_logs = [x for x in logs if x.get("error")]

    positive_feedback = sum(
        1 for x in logs if x.get("feedback") == 1
    )
    negative_feedback = sum(
        1 for x in logs if x.get("feedback") == 0
    )

    feedback_count = positive_feedback + negative_feedback
    feedback_rate = (
        feedback_count / total_queries * 100 if total_queries else 0
    )

    st.subheader("1. Core Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Queries", total_queries)
    col2.metric("Successful", len(successful_logs))
    col3.metric("Failed", len(failed_logs))
    col4.metric("Feedback Rate", f"{feedback_rate:.1f}%")

    st.markdown("---")

    st.subheader("2. Query Volume Over Time")

    if logs:
        query_dates = {}

        for entry in logs:
            date = entry["timestamp"][:10]
            query_dates[date] = query_dates.get(date, 0) + 1

        st.line_chart(query_dates)
    else:
        st.info("No query data yet.")

    st.subheader("3. Response Latency")

    latency_values = [
        x["total_latency_ms"]
        for x in successful_logs
        if x.get("total_latency_ms", 0) > 0
    ]

    if latency_values:
        st.line_chart(latency_values)
    else:
        st.info("No latency data yet.")

    st.subheader("4. Retrieval Pipeline Latency")

    if successful_logs:
        retrieval_data = {
            "Embedding": [
                x.get("embedding_latency_ms", 0) for x in successful_logs
            ],
            "BM25": [
                x.get("bm25_latency_ms", 0) for x in successful_logs
            ],
            "Reranking": [
                x.get("rerank_latency_ms", 0) for x in successful_logs
            ],
        }

        st.line_chart(retrieval_data)
    else:
        st.info("No retrieval telemetry yet.")

    st.subheader("5. User Feedback")

    if feedback_count:
        feedback_data = {
            "Positive 👍": [positive_feedback],
            "Negative 👎": [negative_feedback],
        }

        st.bar_chart(feedback_data)
    else:
        st.info("No feedback data yet.")

    st.subheader("6. Retrieved Documents per Query")

    retrieved_counts = [
        x.get("retrieved_documents", 0)
        for x in successful_logs
    ]

    if retrieved_counts:
        st.line_chart(retrieved_counts)
    else:
        st.info("No retrieval data yet.")

    st.subheader("7. Recent Query Telemetry")

    if logs:
        for entry in reversed(logs[-10:]):
            status = "❌" if entry.get("error") else "✅"
            feedback = (
                "👍"
                if entry.get("feedback") == 1
                else "👎"
                if entry.get("feedback") == 0
                else "—"
            )

            latency = entry.get("total_latency_ms", 0)

            st.text(
                f"{status} [{entry['timestamp']}] "
                f"{feedback} {entry['query']} "
                f"({latency:.0f} ms)"
            )
    else:
        st.info("No queries logged yet.")