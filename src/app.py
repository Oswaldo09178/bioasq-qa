"""
BioASQ Conversational QA — Streamlit UI
Phase 5: Medical Practitioner User Study Interface

Run from project root:
    streamlit run src/app.py
"""

import sys
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "utils"))

import streamlit as st

# ===========================================================================
# Page config — must be first Streamlit call
# ===========================================================================
st.set_page_config(
    page_title="BioASQ Clinical QA",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ===========================================================================
# Custom CSS
# ===========================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Header */
.main-header {
    padding: 1.2rem 0 0.5rem 0;
    border-bottom: 1px solid #e0e0e0;
    margin-bottom: 1.5rem;
}
.main-title {
    font-size: 1.3rem;
    font-weight: 500;
    letter-spacing: -0.01em;
    color: #0f1923;
    margin: 0;
}
.main-subtitle {
    font-size: 0.8rem;
    color: #6b7280;
    margin: 2px 0 0 0;
    font-family: 'IBM Plex Mono', monospace;
}

/* Chat messages */
.user-msg {
    background: #f0f4ff;
    border-left: 3px solid #3b82f6;
    padding: 0.9rem 1rem;
    border-radius: 0 6px 6px 0;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    color: #1e293b;
}
.assistant-msg {
    background: #f8fafc;
    border-left: 3px solid #10b981;
    padding: 0.9rem 1rem;
    border-radius: 0 6px 6px 0;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    color: #1e293b;
    line-height: 1.65;
}
.assistant-msg.flagged {
    border-left-color: #f59e0b;
    background: #fffbeb;
}

/* Metadata badges */
.meta-row {
    display: flex;
    gap: 8px;
    margin-top: 6px;
    flex-wrap: wrap;
}
.badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    padding: 2px 8px;
    border-radius: 3px;
    font-weight: 500;
}
.badge-type  { background: #e0e7ff; color: #3730a3; }
.badge-ok    { background: #d1fae5; color: #065f46; }
.badge-warn  { background: #fef3c7; color: #92400e; }
.badge-time  { background: #f1f5f9; color: #475569; }

/* Evidence panel */
.evidence-title {
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7280;
    margin: 1rem 0 0.4rem 0;
}
.evidence-item {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 0.65rem 0.75rem;
    margin-bottom: 0.5rem;
    font-size: 0.8rem;
    color: #334155;
    line-height: 1.55;
}
.evidence-pmid {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #3b82f6;
    margin-bottom: 4px;
}

/* System config pill */
.config-pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    background: #f1f5f9;
    color: #475569;
    padding: 3px 10px;
    border-radius: 3px;
    display: inline-block;
    margin-bottom: 1rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f1923;
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    color: #94a3b8 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

/* Input */
.stTextInput > div > div > input {
    border-radius: 4px;
    border: 1px solid #e2e8f0;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.9rem;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #94a3b8;
}
.empty-state-icon {
    font-size: 2.5rem;
    margin-bottom: 0.75rem;
}
.empty-state-text {
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# Session state initialization
# ===========================================================================
def init_session():
    if "session_id"    not in st.session_state:
        st.session_state.session_id    = str(uuid.uuid4())
    if "messages"      not in st.session_state:
        st.session_state.messages      = []   # [{role, content, meta}]
    if "rag_system"    not in st.session_state:
        st.session_state.rag_system    = None
    if "system_ready"  not in st.session_state:
        st.session_state.system_ready  = False
    if "last_evidence" not in st.session_state:
        st.session_state.last_evidence = []

init_session()


# ===========================================================================
# Sidebar — system configuration
# ===========================================================================
# ===========================================================================
# Default system config — practitioners never see or change these
# Best-performing values from experiments
# To run in research mode: streamlit run src/app.py -- --research
# ===========================================================================
import sys as _sys
RESEARCH_MODE = "--research" in _sys.argv

DEFAULT_RETRIEVER = "none"
DEFAULT_GENERATOR = "gemini"
DEFAULT_K         = 5
DEFAULT_DATA      = str(PROJECT_ROOT / "data" / "BioASQ-training14b" / "training14b.json")

with st.sidebar:
    if RESEARCH_MODE:
        st.markdown("### ⚙️ Research Mode")
        st.markdown("---")

        DEFAULT_RETRIEVER = st.selectbox(
            "Retriever",
            ["hybrid", "dense", "bm25", "none"],
            index=3,
        )
        DEFAULT_GENERATOR = st.selectbox(
            "Generator",
            ["gemini", "gpt4", "medgemma"],
            index=0,
        )
        DEFAULT_K = st.slider("Top-K documents", min_value=1, max_value=20, value=5)
        DEFAULT_DATA = st.text_input("Dataset path", value=DEFAULT_DATA)
        st.markdown("---")
    else:
        st.markdown("### 🧬 BioASQ Clinical QA")
        st.markdown(
            "<div style='font-size:0.75rem;color:#94a3b8;'>Carnegie Mellon University</div>",
            unsafe_allow_html=True
        )
        st.markdown("---")

    initialize = st.button("Start session", use_container_width=True)

    if initialize:
        with st.spinner("Starting up..."):
            try:
                from rag_system import BioASQRAGSystem
                from data_utils import load_bioasq_dataset, parse_question

                system = BioASQRAGSystem(
                    retriever=DEFAULT_RETRIEVER,
                    generator=DEFAULT_GENERATOR,
                    k=DEFAULT_K,
                )
                system.load_generator()

                if DEFAULT_RETRIEVER != "none":
                    raw_qs    = load_bioasq_dataset(DEFAULT_DATA)
                    questions = [parse_question(q) for q in raw_qs]
                    system.index_corpus(questions)

                st.session_state.rag_system    = system
                st.session_state.system_ready  = True
                st.session_state.messages      = []
                st.session_state.session_id    = str(uuid.uuid4())
                st.session_state.last_evidence = []

                st.success("Ready.")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")

    if st.button("New conversation", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.session_id    = str(uuid.uuid4())
        st.session_state.last_evidence = []
        if st.session_state.rag_system:
            st.session_state.rag_system.reset_session(st.session_state.session_id)
        st.rerun()

    if st.session_state.system_ready and RESEARCH_MODE:
        sys_obj = st.session_state.rag_system
        st.markdown("---")
        st.markdown(f"""
        <div style='font-size:0.7rem; color:#94a3b8; font-family:monospace;'>
        retriever: {sys_obj.retriever_type}<br>
        generator: {sys_obj.generator_type}<br>
        k: {sys_obj.k}<br>
        session: {st.session_state.session_id[:8]}...
        </div>
        """, unsafe_allow_html=True)


# ===========================================================================
# Main layout — chat + evidence panel
# ===========================================================================
chat_col, evidence_col = st.columns([2, 1], gap="large")

with chat_col:
    st.markdown("""
    <div class='main-header'>
        <p class='main-title'>Conversational Biomedical QA</p>
        <p class='main-subtitle'>Multi-turn clinical evidence retrieval · Carnegie Mellon University</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Chat history ---
    if not st.session_state.messages:
        st.markdown("""
        <div class='empty-state'>
            <div class='empty-state-icon'>🔬</div>
            <div class='empty-state-text'>
                Initialize the system in the sidebar, then ask a clinical question.<br>
                <span style='font-size:0.8rem;color:#cbd5e1;'>
                e.g. "What genes are involved in Hirschsprung disease?"
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f"<div class='user-msg'>{msg['content']}</div>",
                    unsafe_allow_html=True
                )
            else:
                meta     = msg.get("meta", {})
                flagged  = meta.get("flagged", False)
                qtype    = meta.get("question_type", "")
                latency  = meta.get("latency_s", 0)
                grounded = meta.get("grounded", False)

                # Render content as plain markdown (safe — no raw HTML injection)
                border_color = "#f59e0b" if flagged else "#10b981"
                st.markdown(
                    f"<div style='border-left:3px solid {border_color};"
                    f"background:#f8fafc;padding:0.9rem 1rem;"
                    f"border-radius:0 6px 6px 0;margin:0.8rem 0;'></div>",
                    unsafe_allow_html=True
                )
                st.markdown(msg["content"])

                # Badges rendered separately — no user content inside HTML
                ground_badge = (
                    "<span class='badge badge-ok'>grounded</span>"
                    if grounded else
                    "<span class='badge badge-warn'>unverified</span>"
                )
                flag_badge = (
                    "<span class='badge badge-warn'>⚠ flagged</span>"
                    if flagged else ""
                )
                st.markdown(
                    f"<div class='meta-row'>"
                    f"<span class='badge badge-type'>{qtype}</span>"
                    f"{ground_badge}"
                    f"{flag_badge}"
                    f"<span class='badge badge-time'>{latency:.2f}s</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    # --- Input ---
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                "Ask a clinical question",
                placeholder="e.g. What are the contraindications for that drug?",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted and user_input.strip():
        if not st.session_state.system_ready:
            st.warning("Please initialize the system first using the sidebar.")
        else:
            # Add user message
            st.session_state.messages.append({
                "role":    "user",
                "content": user_input.strip(),
            })

            # Run inference
            with st.spinner("Searching evidence and generating answer..."):
                try:
                    system   = st.session_state.rag_system
                    question = {
                        "id":   f"ui_{st.session_state.session_id}_{int(time.time())}",
                        "body": user_input.strip(),
                        "type": "summary",
                    }
                    pred = system.answer(
                        question,
                        session_id=st.session_state.session_id,
                    )

                    answer = pred["answer"]
                    if isinstance(answer, list):
                        answer = "\n".join(f"• {a}" for a in answer)

                    # Store evidence for the right panel
                    st.session_state.last_evidence = pred.get("retrieved_snippets", [])

                    # Add assistant message with metadata
                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": answer,
                        "meta":    {
                            "question_type": pred.get("question_type", "summary"),
                            "latency_s":     pred.get("latency_s", 0),
                            "grounded":      pred.get("grounded", False),
                            "flagged":       pred.get("flagged", False),
                        },
                    })

                except Exception as e:
                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": f"Error generating answer: {e}",
                        "meta":    {},
                    })

            st.rerun()


# ===========================================================================
# Evidence panel
# ===========================================================================
with evidence_col:
    st.markdown(
        "<div style='height:4rem'></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='evidence-title'>Retrieved evidence</div>",
        unsafe_allow_html=True
    )

    if not st.session_state.last_evidence:
        st.markdown(
            "<div style='font-size:0.8rem;color:#94a3b8;padding:0.5rem 0;'>"
            "Evidence snippets will appear here after each query."
            "</div>",
            unsafe_allow_html=True
        )
    else:
        for i, doc in enumerate(st.session_state.last_evidence, 1):
            pmid = doc.get("pmid", "")
            text = doc.get("text", "")
            st.markdown(f"""
            <div class='evidence-item'>
                <div class='evidence-pmid'>PMID {pmid} &nbsp;·&nbsp; #{i}</div>
                {text}
            </div>
            """, unsafe_allow_html=True)

        st.markdown(
            f"<div style='font-size:0.7rem;color:#94a3b8;margin-top:4px;'>"
            f"{len(st.session_state.last_evidence)} snippets retrieved</div>",
            unsafe_allow_html=True
        )