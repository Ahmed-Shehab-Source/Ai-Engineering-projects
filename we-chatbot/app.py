"""
WE Business Assistant — a custom-styled RAG chatbot UI over Qdrant + OpenRouter.

Run:
    streamlit run app.py

Required environment variables (set before running, see README below):
    OPENROUTER_API_KEY
    QDRANT_URL
    QDRANT_API_KEY
"""

import os
import re
import time
import html
import requests
import markdown as md
import streamlit as st
from openai import OpenAI, RateLimitError
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# -----------------------------
# CONFIG (from environment)
# -----------------------------
OPENROUTER_API_KEY = "your_openrouter_api_key_here"
QDRANT_URL = "your_qdrant_cluster_url_here"
QDRANT_API_KEY = "your_qdrant_api_key_here"
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "we_chatbot")

# Hardcoded free-model IDs go stale fast (OpenRouter rotates its free lineup
# Hardcoded free-model IDs go stale over time as OpenRouter rotates its free
# lineup, but doing a live fetch on every app start adds real network latency
# (the /models endpoint returns a multi-MB payload of every model OpenRouter
# offers). So: use a short curated list of models verified free & working as
# of Aug 2026 by default (instant startup, no network call), and only hit the
# live API if the person explicitly asks for a refresh via the sidebar button.
VERIFIED_FREE_MODELS_AUG_2026 = [
    "inclusionai/ling-3.0-tiny:free",      # 1.3B active params — very fast
    "poolside/laguna-xs-2.1:free",         # 33B-A3B — fast, coding-capable
    "cohere/north-mini-code:free",         # 30B MoE, 3B active — fast
    "poolside/laguna-s-2.1:free",          # 118B total, 8B active — moderate
    "nvidia/nemotron-3-ultra-550b-a55b:free",  # 55B active — slower fallback
    "openrouter/free",  # OpenRouter's own auto-router as last-ditch fallback
]

# Model IDs that are free but NOT general chat models (moderation/guardrail
# models, image generators, etc.) -- never usable for answering questions.
_EXCLUDE_SUBSTRINGS = ["safety", "guard", "moderation", "-image", "content-safety"]


@st.cache_resource(show_spinner=False)
def fetch_free_chat_models(limit=6):
    """Query OpenRouter's live model list and return up to `limit` fast, free,
    text-in/text-out chat model IDs. Only called on-demand (sidebar refresh
    button), never automatically at startup, to avoid adding network latency
    to every app launch."""
    try:
        resp = requests.get("https://openrouter.ai/api/v1/models", timeout=6)
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception:
        return []

    candidates = []
    for m in data:
        model_id = m.get("id", "")
        if not model_id.endswith(":free"):
            continue
        if any(bad in model_id.lower() for bad in _EXCLUDE_SUBSTRINGS):
            continue
        pricing = m.get("pricing", {})
        if pricing.get("prompt") != "0" or pricing.get("completion") != "0":
            continue
        arch = m.get("architecture", {})
        in_mod = arch.get("input_modalities", [])
        out_mod = arch.get("output_modalities", [])
        if "text" not in in_mod or out_mod != ["text"]:
            continue  # skip image/audio-output or non-text-input models

        reasoning = m.get("reasoning", {}) or {}
        mandatory_reasoning = bool(reasoning.get("mandatory"))
        context_length = m.get("context_length") or 0

        candidates.append({
            "id": model_id,
            "mandatory_reasoning": mandatory_reasoning,
            "context_length": context_length,
        })

    candidates.sort(key=lambda c: (c["mandatory_reasoning"], c["context_length"]))
    return [c["id"] for c in candidates[:limit]]


DEFAULT_MODEL_CHAIN = VERIFIED_FREE_MODELS_AUG_2026

LLM_MODELS = [
    m.strip() for m in os.environ.get("LLM_MODELS", ",".join(DEFAULT_MODEL_CHAIN)).split(",")
    if m.strip()
]


APP_NAME = os.environ.get("APP_NAME", "Aether")

SYSTEM_PROMPT = f"""You are {APP_NAME}, a helpful customer support assistant for WE (Telecom Egypt).
Answer the user's question using ONLY the information in the provided context.
If the context does not contain the answer, say you don't have that information
and suggest contacting WE customer service.

LANGUAGE RULE (follow strictly): Detect the language of the user's QUESTION only —
ignore the language of the retrieved context. If the question is in English, your
entire answer must be in English, even if all retrieved context is in Arabic
(translate the relevant facts yourself). If the question is in Arabic, answer in
Arabic. Never mix languages in one answer.

Be thorough and include the specific, useful details from the context
(prices, package names, MB/unit amounts, features) rather than a high-level
summary — the user wants the actual numbers and facts, not just an overview.
Avoid markdown headers (#, ##) for a chat answer — use short bold labels or
plain sentences instead. Use a real markdown table (with a header row and a
|---|---| separator row) when comparing multiple packages side by side.
"""


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title=APP_NAME,
    page_icon="📡",
    layout="centered",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Custom CSS — ditch the default Streamlit chat look
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header {visibility: hidden;}

.stApp {
    background: radial-gradient(circle at 15% 10%, #1b1035 0%, #0a0714 45%, #060410 100%);
}

/* Header banner */
.we-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 22px;
    margin-bottom: 22px;
    border-radius: 18px;
    background: linear-gradient(120deg, rgba(124,58,237,0.35), rgba(236,72,153,0.25));
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 30px rgba(124,58,237,0.15);
}
.we-header .logo {
    width: 46px; height: 46px;
    border-radius: 12px;
    background: linear-gradient(135deg, #7c3aed, #ec4899);
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    flex-shrink: 0;
}
.we-header .titles h1 {
    font-family: 'Space Grotesk', sans-serif;
    color: #fff;
    font-size: 21px;
    margin: 0;
    letter-spacing: 0.3px;
}
.we-header .titles p {
    color: rgba(255,255,255,0.55);
    font-size: 12.5px;
    margin: 2px 0 0 0;
}

/* Chat scroll area */
.chat-wrap {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding-bottom: 10px;
}

/* Bubble base */
.bubble {
    max-width: 78%;
    padding: 10px 15px;
    border-radius: 16px;
    font-size: 14.5px;
    line-height: 1.45;
    white-space: pre-wrap;
    word-wrap: break-word;
    animation: fadein 0.25s ease-out;
}
@keyframes fadein {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

.row { display: flex; width: 100%; }
.row.user { justify-content: flex-end; }
.row.bot { justify-content: flex-start; }

.bubble.user {
    background: linear-gradient(135deg, #7c3aed, #4c1d95);
    color: #fff;
    border-bottom-right-radius: 4px;
}
.bubble.bot {
    background: rgba(255,255,255,0.06);
    color: #ece9f7;
    border: 1px solid rgba(255,255,255,0.08);
    border-bottom-left-radius: 4px;
}

.meta-tag {
    display: block;
    font-size: 10.5px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    opacity: 0.5;
    margin-bottom: 4px;
}

/* Sources */
.sources-box {
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px dashed rgba(255,255,255,0.15);
    font-size: 11.5px;
    color: rgba(255,255,255,0.55);
}
.source-pill {
    display: inline-block;
    background: rgba(124,58,237,0.25);
    border: 1px solid rgba(124,58,237,0.4);
    color: #e9d5ff;
    padding: 2px 9px;
    border-radius: 999px;
    margin: 3px 4px 0 0;
    font-size: 10.5px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d0a1c;
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] * { color: #e5e0f5 !important; }

/* Input box override */
.stChatInput, div[data-testid="stChatInput"] {
    border-radius: 14px !important;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: rgba(255,255,255,0.4);
}
.empty-state .big {
    font-size: 40px;
    margin-bottom: 8px;
}

/* Rendered markdown inside bot bubbles */
.bubble.bot p { margin: 0 0 5px 0; }
.bubble.bot p:first-child { margin-top: 0; }
.bubble.bot p:last-child { margin-bottom: 0; }
.bubble.bot h1, .bubble.bot h2, .bubble.bot h3,
.bubble.bot h4, .bubble.bot h5, .bubble.bot h6 {
    color: #fff;
    font-family: 'Space Grotesk', sans-serif;
    margin: 10px 0 4px 0;
    line-height: 1.3;
}
.bubble.bot h1:first-child, .bubble.bot h2:first-child,
.bubble.bot h3:first-child { margin-top: 0; }
.bubble.bot h1 { font-size: 15.5px; }
.bubble.bot h2 { font-size: 15px; }
.bubble.bot h3, .bubble.bot h4, .bubble.bot h5, .bubble.bot h6 { font-size: 14.5px; }
.bubble.bot strong { color: #fff; }
.bubble.bot ul, .bubble.bot ol { margin: 3px 0 5px 20px; padding: 0; }
.bubble.bot li { margin-bottom: 1px; }
.bubble.bot table {
    border-collapse: collapse;
    width: 100%;
    margin: 5px 0;
    font-size: 13px;
    direction: rtl;
}
.bubble.bot th, .bubble.bot td {
    border: 1px solid rgba(255,255,255,0.15);
    padding: 4px 9px;
    text-align: center;
    line-height: 1.3;
}
.bubble.bot th {
    background: rgba(124,58,237,0.25);
    color: #fff;
}
.bubble.bot code {
    background: rgba(255,255,255,0.1);
    padding: 1px 5px;
    border-radius: 4px;
    font-size: 13px;
}
.bubble.bot hr { margin: 6px 0; border-color: rgba(255,255,255,0.1); }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Cached backend connections
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_backend():
    if not (OPENROUTER_API_KEY and QDRANT_URL and QDRANT_API_KEY):
        return None
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
    )
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        timeout=12.0,  # fail fast on a slow/unresponsive free model instead of hanging
        max_retries=0,  # we handle retries/fallback ourselves across the model chain
    )
    return vector_store, client


def retrieve(vector_store, query, k=5):
    results = vector_store.similarity_search_with_score(query, k=k)
    return [
        {"content": doc.page_content, "metadata": doc.metadata, "score": score}
        for doc, score in results
    ]


def format_context(chunks):
    return "\n\n---\n\n".join(c["content"] for c in chunks)


def normalize_markdown(text: str) -> str:
    """Fix common malformed-markdown issues from free LLMs before rendering:
    - Insert a blank line before a pipe-table if the model ran it directly
      after a paragraph (python-markdown's table extension needs a blank
      line to recognize a new block, otherwise it renders as plain text).
    """
    if not text:
        return text
    lines = text.split("\n")
    fixed = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        is_table_row = stripped.startswith("|") and stripped.endswith("|")
        prev_blank = (i == 0) or (lines[i - 1].strip() == "")
        prev_is_table_row = (
            i > 0 and lines[i - 1].strip().startswith("|") and lines[i - 1].strip().endswith("|")
        )
        if is_table_row and not prev_blank and not prev_is_table_row:
            fixed.append("")  # insert the missing blank line
        fixed.append(line)
    return "\n".join(fixed)


def clean_answer(text: str):
    if not text or not text.strip():
        return None
    lines = text.splitlines()
    cleaned = []
    meta_pattern = re.compile(
        r"^\s*(\[?\s*(user\s*safety|safety|moderation|content\s*flag)\s*\]?\s*:\s*\w+\s*\]?)\s*$",
        re.IGNORECASE,
    )
    for line in lines:
        if meta_pattern.match(line):
            continue
        cleaned.append(line)
    result = "\n".join(cleaned).strip()
    return result or None


def ask(vector_store, client, query, history=None, k=5, history_turns=1, models=None):
    """
    history: list of {"role": "user"/"assistant", "content": str} from prior turns
    (most recent last), used both to give the LLM conversation memory and to
    build a better retrieval query for vague follow-ups like "tell me more".
    """
    history = history or []

    # Build a retrieval query that folds in recent context, so follow-ups
    # like "more details?" don't search on those two words alone.
    recent_for_search = history[-2:] if history else []
    search_query_parts = [h["content"] for h in recent_for_search] + [query]
    search_query = "\n".join(search_query_parts)

    chunks = retrieve(vector_store, search_query, k=k)
    context = format_context(chunks)
    user_prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n"

    # Give the model actual conversation memory (last N exchanges)
    trimmed_history = history[-(history_turns * 2):]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(trimmed_history)
    messages.append({"role": "user", "content": user_prompt})

    last_error = None
    for model in (models or LLM_MODELS):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            raw_answer = response.choices[0].message.content
            answer = clean_answer(raw_answer)
            if answer is None:
                # Model returned nothing usable (e.g. only a safety/meta line) — try next model
                last_error = RuntimeError(f"{model} returned no usable content")
                continue
            return answer, chunks, model
        except RateLimitError as e:
            last_error = e
            continue  # rate-limited -> skip straight to next model, no point retrying same one
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"All models exhausted. Last error: {last_error}")


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    k = st.slider("Chunks to retrieve (k)", 1, 10, 5)
    show_sources = st.toggle("Show sources", value=True)
    st.markdown("---")
    st.markdown("### 🔑 Connection status")
    ok = bool(OPENROUTER_API_KEY and QDRANT_URL and QDRANT_API_KEY)
    st.markdown(f"{'🟢' if ok else '🔴'} Env vars loaded" )
    st.caption(f"Collection: `{COLLECTION_NAME}`")

    if "active_models" not in st.session_state:
        st.session_state.active_models = LLM_MODELS

    st.caption(f"Model chain: {' → '.join(st.session_state.active_models)}")
    if st.button("🔄 Refresh free-model list from OpenRouter", use_container_width=True):
        with st.spinner("Checking OpenRouter's live free-model list…"):
            fetch_free_chat_models.clear()  # bust the cached result
            live = fetch_free_chat_models(limit=6)
        if live:
            st.session_state.active_models = live + ["openrouter/free"]
            st.success(f"Updated: {len(live)} live free models found")
        else:
            st.warning("Couldn't reach OpenRouter — kept current model list")
        st.rerun()

    st.markdown("---")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="we-header">'
    '<div class="logo">📡</div>'
    '<div class="titles">'
    f'<h1>{APP_NAME}</h1>'
    "<p>Ask me anything about WE's business services — Arabic or English</p>"
    '</div></div>',
    unsafe_allow_html=True,
)

# -----------------------------
# State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: role, content, sources(optional)

backend = get_backend()

if backend is None:
    st.error(
        "Missing API keys. Set OPENROUTER_API_KEY, QDRANT_URL and QDRANT_API_KEY "
        "as environment variables before launching the app (see README)."
    )
    st.stop()

vector_store, client = backend

# -----------------------------
# Render chat history
# -----------------------------
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.markdown(
            '<div class="empty-state">'
            '<div class="big">💬</div>'
            '<div>Start the conversation — try "How do I renew my internet package?"</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        role = msg["role"]
        css_role = "user" if role == "user" else "bot"  # CSS classes are .user/.bot, not "assistant"
        if role == "user":
            content = html.escape(msg["content"]).replace("\n", "<br>")
        else:
            # Render markdown (bold, tables, lists, etc.) returned by the LLM
            content = md.markdown(
                normalize_markdown(msg["content"]), extensions=["extra", "sane_lists"]
            )
        tag = "You" if role == "user" else APP_NAME
        sources_html = ""
        if role == "assistant" and show_sources and msg.get("sources"):
            pills = "".join(
                f'<span class="source-pill">{html.escape(str(s["metadata"].get("source", "unknown")))} · {s["score"]:.2f}</span>'
                for s in msg["sources"]
            )
            model_tag = f' · <em>{html.escape(msg.get("model", ""))}</em>' if msg.get("model") else ""
            sources_html = f'<div class="sources-box">Sources: {pills}{model_tag}</div>'
        row_html = (
            f'<div class="row {css_role}">'
            f'<div class="bubble {css_role}">'
            f'<span class="meta-tag">{tag}</span>'
            f'{content}'
            f'{sources_html}'
            f'</div></div>'
        )
        st.markdown(row_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Input
# -----------------------------
query = st.chat_input("Type your question in Arabic or English…")

if query:
    # Snapshot prior turns (before appending the new question) for memory/context
    prior_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m["role"] in ("user", "assistant")
    ]

    st.session_state.messages.append({"role": "user", "content": query})
    with st.spinner("Searching knowledge base and thinking…"):
        try:
            answer, chunks, used_model = ask(
                vector_store, client, query, history=prior_history, k=k,
                models=st.session_state.active_models,
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": chunks, "model": used_model}
            )
        except Exception as e:
            st.session_state.messages.append(
                {"role": "assistant", "content": f"⚠️ Error: {e}", "sources": []}
            )
    st.rerun()
