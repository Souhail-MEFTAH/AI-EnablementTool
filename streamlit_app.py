import os
import base64
import asyncio

import streamlit as st
from dotenv import load_dotenv

from rag_agent import run_rag_agent
from utils import get_chroma_client, get_or_create_collection

# ─── 1) PAGE CONFIG (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Vantiq Use Case Assistant",
    layout="wide",
)

load_dotenv()

# ─── 2) WARM UP EMBEDDINGS & DB ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_db_and_embeddings():
    client = get_chroma_client("./chroma_db")
    get_or_create_collection(
        client,
        "docs",
        embedding_model_name="all-MiniLM-L6-v2"
    )
    return client

_ = init_db_and_embeddings()

# ─── 3) SESSION STATE ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role":.., "content":..}

if "current_input" not in st.session_state:
    st.session_state.current_input = None

if "last_processed" not in st.session_state:
    st.session_state.last_processed = None

# ─── 4) HEADER / LOGO ────────────────────────────────────────────────────────────
logo_path = os.path.join(os.path.dirname(__file__), "vantiq_logo.png")
if os.path.exists(logo_path):
    img = open(logo_path, "rb").read()
    b64 = base64.b64encode(img).decode()
    st.markdown(
        f"""<div style="text-align:center; margin-bottom:1rem;">
               <img src="data:image/png;base64,{b64}" width="120" />
           </div>""",
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div style="text-align:center;">
      <h1 style="margin:0;">Vantiq Use Case Assistant</h1>
      <p style="margin:0.5rem 0 1.5rem; font-size:1.1rem;">
        Your real-time guide to event-driven, AI-powered applications on Vantiq.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── 5) SHOW EXISTING CHAT HISTORY ────────────────────────────────────────────────
for msg in st.session_state.history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# ─── 6) SUGGESTION BUTTONS (only before the first query) ──────────────────────────
if not st.session_state.history:
    suggestions = [
        "What are examples of agentic ai being used for healthcare operations?",
        "How do I build a public safety alert system with Vantiq?",
        "How do I integrate Vantiq with OpenAI models?",
    ]
    _, c1, c2, c3, _ = st.columns([1,3,3,3,1])
    for col, text, key in zip((c1, c2, c3), suggestions, range(3)):
        if col.button(text, key=f"sugg_{key}", use_container_width=True):
            st.session_state.current_input = text

# ─── 7) CHAT INPUT BAR ───────────────────────────────────────────────────────────
user_text = st.chat_input("What do you want to know?")  # always visible
if user_text:
    st.session_state.current_input = user_text

# ─── 8) PROCESS NEW INPUT RIGHT AWAY ─────────────────────────────────────────────
ci = st.session_state.current_input
lp = st.session_state.last_processed

if ci and ci != lp:
    # 8a) record & display user message
    st.session_state.history.append({"role": "user", "content": ci})
    with st.chat_message("user"):
        st.markdown(ci)

    # 8b) show global spinner while we fetch the answer
    with st.spinner("Thinking..."):
        # if there’s prior convo, prefix into a single prompt
        if st.session_state.history[:-1]:
            hist = "\n".join(
                f"User: {h['content']}\nAssistant: {a['content']}"
                for h, a in zip(
                    st.session_state.history[0::2],
                    st.session_state.history[1::2],
                )
            )
            prompt = f"{hist}\nUser: {ci}"
        else:
            prompt = ci

        answer = asyncio.run(
            run_rag_agent(
                question=prompt,
                collection_name="docs",
                db_directory="./chroma_db",
                embedding_model="all-MiniLM-L6-v2",
                n_results=5,
            )
        )

    # 8c) now render the assistant bubble *once*
    st.session_state.history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    # 8d) mark done
    st.session_state.last_processed = ci
    st.session_state.current_input = None

# ─── 9) FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; font-size:0.8em;'>© 2025 Vantiq</p>",
    unsafe_allow_html=True,
)
