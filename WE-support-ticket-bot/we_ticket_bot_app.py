"""
we_ticket_bot_app.py - WE Ticket Classification, Summarization & Sentiment.

Run with:
    streamlit run we_ticket_bot_app.py

Expects (from your Colab fine-tunes):
    ./mt5_ticket_model/              (folder - model + tokenizer)
    ./marbertv2_multitask.pt         (state dict)
    ./marbertv2_tokenizer/           (folder - tokenizer)

WHAT CHANGED (per feedback: BERT classifies well, mT5 summarizes decently
but classifies poorly):
  - The primary output is now a HYBRID: category + sentiment come from
    MARBERTv2 (it's the stronger classifier), summary comes from mT5
    (it's the only one of the two that can generate text at all). This
    is just "take the best tool for each sub-task" rather than forcing
    one model to do everything.
  - Added an optional THIRD column: a free-tier hosted LLM (Groq or
    OpenRouter) called via a single prompt asking for the same
    category/sentiment/summary JSON. This is a sanity-check comparison
    against your two locally fine-tuned models - costs nothing on the
    free tier, no local GPU needed for this column since it's an API call.
  - Raw/individual model outputs (MARBERTv2-only, mT5-only) are still
    shown in an expandable section for debugging, not as the headline.
"""

import os
import re
import io
import json
import requests
import torch
import torch.nn as nn
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

# ─────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────
CATEGORY_LABELS = ["Billing", "Network", "Technical Support", "Sales/Plans", "Other"]
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]
PREFIX = "classify and summarize ticket: "

MT5_DIR = "mt5_ticket_model"
MARBERT_WEIGHTS = "marbertv2_multitask.pt"
MARBERT_TOKENIZER_DIR = "marbertv2_tokenizer"
MARBERT_BASE_MODEL = "UBC-NLP/MARBERTv2"

SENTIMENT_COLORS = {"Positive": "🟢", "Neutral": "⚪", "Negative": "🔴", "UNCERTAIN": "🟡"}

PARSE_RE = re.compile(
    r"category:\s*(?P<category>.*?)\s*\|\s*sentiment:\s*(?P<sentiment>.*?)\s*\|\s*summary:\s*(?P<summary>.*)",
    re.IGNORECASE | re.DOTALL,
)

# ── API keys: fill these in directly (do NOT commit real keys to a public repo) ──
GROQ_API_KEY = "PASTE_YOUR_GROQ_API_KEY_HERE"
OPENROUTER_API_KEY = "PASTE_YOUR_OPENROUTER_API_KEY_HERE"

PROVIDER_ENDPOINTS = {
    "Groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "default_model": "llama-3.1-8b-instant",
        "api_key": GROQ_API_KEY,
    },
    "OpenRouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "default_model": "meta-llama/llama-3.1-8b-instruct:free",
        "api_key": OPENROUTER_API_KEY,
    },
}

# IMPORTANT: this is a TICKET SUMMARIZER, not a customer-support chatbot.
# The model must describe/log what the customer said - never answer them,
# never offer help, never address them in 2nd person ("you can...").
API_SYSTEM_PROMPT = (
    "You are a backend ticket-triage tool for an Egyptian telecom company (WE), used internally "
    "by support agents. You are NOT a chatbot and you must NEVER talk to, help, or answer the "
    "customer. You will be given ONE raw customer ticket, possibly in English, Arabic, "
    "Franco-Arabic, or a mix, sometimes with typos or emojis.\n\n"
    "Your ONLY job is to log THIS SPECIFIC ticket for an agent - never invent a different "
    "scenario, never answer a question the customer didn't ask, and never respond as if you were "
    "talking to the customer. Base the summary ONLY on words actually present in the ticket.\n\n"
    "Examples:\n"
    '  Ticket: "3andy moshkela fel fatoora, atkhasam maraten" -> '
    '{"category": "Billing", "sentiment": "Negative", '
    '"summary": "العميل يشتكي من خصم مبلغ الفاتورة مرتين."}\n'
    '  Ticket: "can i update my billing address and credit card details from the app or do i '
    'need to visit a branch?" -> '
    '{"category": "Billing", "sentiment": "Neutral", '
    '"summary": "Customer asks whether billing address and card details can be updated via the '
    'app or require a branch visit."}\n'
    '  Ticket: "i recharged 100 EGP via InstaPay w rasidi msalsh 3al account \\ud83d\\ude14" -> '
    '{"category": "Billing", "sentiment": "Negative", '
    '"summary": "العميل يشتكي من انه شحن 100 جنيه عن طريق InstaPay والرصيد لم يصل لحسابه."}\n\n'
    "Rules:\n"
    f"1. category must be exactly one of: {', '.join(CATEGORY_LABELS)}.\n"
    f"2. sentiment must be exactly one of: {', '.join(SENTIMENT_LABELS)}.\n"
    "3. summary must be a 1-2 sentence THIRD-PERSON description of what the customer said or "
    "asked - e.g. 'Customer reports X' / 'العميل يسأل عن Y' - written FLUENTLY and "
    "GRAMMATICALLY in the SAME language as the ticket (pure Arabic for Arabic/Franco-Arabic "
    "tickets, pure English for English tickets). Never mix English words into an Arabic "
    "summary except for brand/product names (e.g. InstaPay, WE, VISA).\n"
    "4. NEVER answer the customer's question, NEVER give instructions, NEVER say 'you can...', "
    "NEVER address the customer directly - you are logging the ticket, not resolving it.\n"
    "5. NEVER fabricate details, amounts, or topics that are not in the ticket text.\n"
    "6. Respond with ONLY a single valid JSON object with exactly the keys: category, sentiment, "
    "summary. No markdown fences, no preamble, no trailing text, no comments. All string values "
    "must be properly JSON-escaped (escape internal quotes, no literal newlines inside strings)."
)


# ─────────────────────────────────────────────────────────────────────────
# Local model loading (cached)
# ─────────────────────────────────────────────────────────────────────────
class MultiTaskMARBERT(nn.Module):
    def __init__(self, model_name, n_cat, n_sent, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.cat_head = nn.Linear(hidden, n_cat)
        self.sent_head = nn.Linear(hidden, n_sent)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0])
        return self.cat_head(pooled), self.sent_head(pooled)


@st.cache_resource(show_spinner="Loading mT5 model...")
def load_mt5():
    if not os.path.isdir(MT5_DIR):
        return None, None
    tok = AutoTokenizer.from_pretrained(MT5_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MT5_DIR)
    model.eval()
    return tok, model


@st.cache_resource(show_spinner="Loading MARBERTv2 model...")
def load_marbert():
    if not (os.path.exists(MARBERT_WEIGHTS) and os.path.isdir(MARBERT_TOKENIZER_DIR)):
        return None, None
    tok = AutoTokenizer.from_pretrained(MARBERT_TOKENIZER_DIR)
    model = MultiTaskMARBERT(MARBERT_BASE_MODEL, len(CATEGORY_LABELS), len(SENTIMENT_LABELS))
    model.load_state_dict(torch.load(MARBERT_WEIGHTS, map_location="cpu"))
    model.eval()
    return tok, model


# ─────────────────────────────────────────────────────────────────────────
# Inference helpers - local models
# ─────────────────────────────────────────────────────────────────────────
def parse_mt5_output(raw_text):
    m = PARSE_RE.search(raw_text)
    if not m:
        return {"category": "UNCERTAIN", "sentiment": "UNCERTAIN", "summary": raw_text.strip()}
    cat = m.group("category").strip()
    sent = m.group("sentiment").strip()
    summary = m.group("summary").strip()
    cat_match = next((c for c in CATEGORY_LABELS if c.lower() == cat.lower()), "UNCERTAIN")
    sent_match = next((s for s in SENTIMENT_LABELS if s.lower() == sent.lower()), "UNCERTAIN")
    return {"category": cat_match, "sentiment": sent_match, "summary": summary}


def run_mt5(tok, model, text):
    inputs = tok(PREFIX + text, return_tensors="pt", truncation=True, max_length=96)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=80, num_beams=4, no_repeat_ngram_size=3)
    raw = tok.decode(output_ids[0], skip_special_tokens=True)
    return parse_mt5_output(raw)


def run_marbert(tok, model, text):
    inputs = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=96)
    with torch.no_grad():
        cat_logits, sent_logits = model(inputs["input_ids"], inputs["attention_mask"])
    cat_probs = torch.softmax(cat_logits, dim=1)[0]
    sent_probs = torch.softmax(sent_logits, dim=1)[0]
    cat_idx = int(cat_probs.argmax())
    sent_idx = int(sent_probs.argmax())
    return {
        "category": CATEGORY_LABELS[cat_idx],
        "category_conf": round(float(cat_probs[cat_idx]), 2),
        "sentiment": SENTIMENT_LABELS[sent_idx],
        "sentiment_conf": round(float(sent_probs[sent_idx]), 2),
    }


def run_hybrid(marbert_result, mt5_result):
    """Category + sentiment from MARBERTv2 (stronger classifier), summary from mT5
    (only one of the two that can actually generate text)."""
    return {
        "category": marbert_result["category"],
        "category_conf": marbert_result["category_conf"],
        "sentiment": marbert_result["sentiment"],
        "sentiment_conf": marbert_result["sentiment_conf"],
        "summary": mt5_result["summary"],
    }


# ─────────────────────────────────────────────────────────────────────────
# Inference helper - free-tier hosted LLM API (Groq / OpenRouter)
# ─────────────────────────────────────────────────────────────────────────
# Regex fallback: pulls category/sentiment/summary out of near-JSON text even if the
# JSON itself is malformed (unescaped quotes, stray text, trailing commas, etc.)
_API_FIELD_RE = re.compile(
    r'"category"\s*:\s*"(?P<category>[^"]*)".*?'
    r'"sentiment"\s*:\s*"(?P<sentiment>[^"]*)".*?'
    r'"summary"\s*:\s*"(?P<summary>.*?)"\s*[},]',
    re.IGNORECASE | re.DOTALL,
)


def _strip_code_fences(content):
    return re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.MULTILINE).strip()


def _parse_api_json(content):
    """Try strict JSON first, then a regex-based best-effort extraction. Raises on total failure."""
    content_clean = _strip_code_fences(content)
    try:
        return json.loads(content_clean)
    except json.JSONDecodeError:
        pass
    # Some models wrap valid JSON in extra prose - grab the outermost {...} block and retry
    brace_match = re.search(r"\{.*\}", content_clean, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    # Last resort: regex field extraction (handles unescaped internal quotes/newlines)
    m = _API_FIELD_RE.search(content_clean)
    if m:
        return {
            "category": m.group("category"),
            "sentiment": m.group("sentiment"),
            "summary": m.group("summary"),
        }
    raise ValueError(f"Could not parse JSON or extract fields from model output: {content_clean[:200]!r}")


def _build_user_prompt(text):
    lang = "Arabic" if is_arabic_heuristic(text) else "English/Franco-Arabic"
    return (
        f"Ticket language appears to be: {lang}. Write the summary in that same language "
        f"(pure Arabic if Arabic/Franco-Arabic, pure English if English).\n\n"
        f"Ticket text (log exactly this, nothing else):\n{text}"
    )


def is_arabic_heuristic(text):
    arabic_chars = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
    return arabic_chars > len(text) * 0.15


def _call_llm_once(provider, model_name, text, extra_system_note=None):
    endpoint = PROVIDER_ENDPOINTS[provider]["url"]
    api_key = PROVIDER_ENDPOINTS[provider]["api_key"]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    system_content = API_SYSTEM_PROMPT
    if extra_system_note:
        system_content += "\n\n" + extra_system_note

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": _build_user_prompt(text)},
        ],
        "temperature": 0.0,
        "max_tokens": 400,
    }
    # Groq (and OpenRouter, for models that support it) can enforce valid JSON output directly -
    # this eliminates most of the malformed-JSON failures at the source.
    if provider == "Groq":
        payload["response_format"] = {"type": "json_object"}

    resp = requests.post(endpoint, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_llm_api(provider, model_name, text):
    try:
        raw_content = _call_llm_once(provider, model_name, text)
    except Exception as e:
        return {"category": "ERROR", "sentiment": "ERROR", "summary": "", "error": f"API call failed: {e}"}

    parse_error = None
    parsed = None
    try:
        parsed = _parse_api_json(raw_content)
    except Exception as e:
        parse_error = str(e)

    # One-shot repair retry: if parsing failed, ask the model again with an explicit
    # instruction to fix its formatting, quoting the exact same ticket.
    if parsed is None:
        try:
            repair_note = (
                "IMPORTANT: Your previous response was not valid JSON and was rejected. "
                "Respond again for the SAME ticket, with ONLY one valid JSON object on a single "
                "line: {\"category\": \"...\", \"sentiment\": \"...\", \"summary\": \"...\"}. "
                "Escape any quotes inside the summary as \\\". Do not add any other text."
            )
            raw_content_retry = _call_llm_once(provider, model_name, text, extra_system_note=repair_note)
            parsed = _parse_api_json(raw_content_retry)
            parse_error = None
        except Exception as e:
            return {
                "category": "ERROR",
                "sentiment": "ERROR",
                "summary": "",
                "error": f"JSON parse failed after retry: {parse_error} | retry error: {e}",
            }

    cat = parsed.get("category", "UNCERTAIN")
    sent = parsed.get("sentiment", "UNCERTAIN")
    cat_match = next((c for c in CATEGORY_LABELS if c.lower() == str(cat).strip().lower()), "UNCERTAIN")
    sent_match = next((s for s in SENTIMENT_LABELS if s.lower() == str(sent).strip().lower()), "UNCERTAIN")
    return {
        "category": cat_match,
        "sentiment": sent_match,
        "summary": str(parsed.get("summary", "")).strip(),
        "error": None,
    }


# ─────────────────────────────────────────────────────────────────────────
# Combined per-ticket processing
# ─────────────────────────────────────────────────────────────────────────
def process_ticket(text, mt5_tok, mt5_model, marbert_tok, marbert_model,
                    use_api=False, api_provider=None, api_model_name=None):
    result = {"text": text}

    marbert_result = run_marbert(marbert_tok, marbert_model, text) if marbert_model is not None else None
    mt5_result = run_mt5(mt5_tok, mt5_model, text) if mt5_model is not None else None

    if marbert_result is not None:
        result["marbert_category"] = marbert_result["category"]
        result["marbert_category_conf"] = marbert_result["category_conf"]
        result["marbert_sentiment"] = marbert_result["sentiment"]
        result["marbert_sentiment_conf"] = marbert_result["sentiment_conf"]

    if mt5_result is not None:
        result["mt5_category"] = mt5_result["category"]
        result["mt5_sentiment"] = mt5_result["sentiment"]
        result["mt5_summary"] = mt5_result["summary"]

    if marbert_result is not None and mt5_result is not None:
        hybrid = run_hybrid(marbert_result, mt5_result)
        result["hybrid_category"] = hybrid["category"]
        result["hybrid_sentiment"] = hybrid["sentiment"]
        result["hybrid_summary"] = hybrid["summary"]
        result["needs_review"] = hybrid["category_conf"] < 0.5 or hybrid["sentiment_conf"] < 0.5
    else:
        result["needs_review"] = False

    result["used_api"] = use_api
    if use_api:
        api_result = call_llm_api(api_provider, api_model_name, text)
        result["api_category"] = api_result["category"]
        result["api_sentiment"] = api_result["sentiment"]
        result["api_summary"] = api_result["summary"]
        result["api_error"] = api_result["error"]
        result["api_provider_used"] = api_provider
        result["api_model_used"] = api_model_name

    return result


# ─────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="WE Ticket Bot", page_icon="📶", layout="wide")
st.title("📶 WE Ticket Bot - Classification, Sentiment & Summary")
st.caption(
    "**Hybrid engine** (headline output): category + sentiment from fine-tuned MARBERTv2 "
    "(stronger classifier), summary from fine-tuned mT5-small (only one that can generate text). "
    "Optionally compare against a free-tier hosted LLM (Groq/OpenRouter)."
)

mt5_tok, mt5_model = load_mt5()
marbert_tok, marbert_model = load_marbert()

if mt5_model is None:
    st.error(f"mT5 model not found at `./{MT5_DIR}/`. Run 03_finetune_mt5small.py in Colab and place the folder here.")
if marbert_model is None:
    st.error(
        f"MARBERTv2 model not found (`{MARBERT_WEIGHTS}` / `{MARBERT_TOKENIZER_DIR}/`). "
        "Run 02_finetune_marbertv2.py in Colab and place the files here."
    )

models_ready = (mt5_model is not None) and (marbert_model is not None)

# ── Sidebar: free-tier API comparison setup ─────────────────────────────
# API keys are hardcoded near the top of this file (GROQ_API_KEY / OPENROUTER_API_KEY) -
# no key entry in the UI.
st.sidebar.header("Optional: LLM API comparison")
use_api = st.sidebar.checkbox("Enable Groq/OpenRouter comparison column", value=False)
api_provider = None
api_model_name = None
if use_api:
    api_provider = st.sidebar.selectbox("Provider", list(PROVIDER_ENDPOINTS.keys()))
    api_model_name = st.sidebar.text_input(
        "Model name", value=PROVIDER_ENDPOINTS[api_provider]["default_model"]
    )
    st.sidebar.caption(
        "Groq: console.groq.com (free tier). OpenRouter: openrouter.ai - use a `:free` model suffix "
        "like `meta-llama/llama-3.1-8b-instruct:free` for zero-cost calls."
    )
    configured_key = PROVIDER_ENDPOINTS[api_provider]["api_key"]
    if not configured_key or configured_key.startswith("PASTE_YOUR_"):
        st.sidebar.warning(
            f"Set {api_provider.upper()}_API_KEY near the top of we_ticket_bot_app.py before using this."
        )

tab_single, tab_batch = st.tabs(["Single ticket", "Batch (CSV)"])

# ── Single ticket tab ───────────────────────────────────────────────────
with tab_single:
    text_input = st.text_area(
        "Paste a customer ticket (English, Arabic, Franco-Arabic, or mixed):",
        height=120,
        placeholder="مثال: الانترنت واقع من يومين ومحدش رد عليا...",
    )

    configured_ok = True
    if use_api:
        configured_ok = not PROVIDER_ENDPOINTS[api_provider]["api_key"].startswith("PASTE_YOUR_")
    can_run = models_ready and (not use_api or configured_ok)

    if st.button("Analyze ticket", type="primary", disabled=not can_run):
        if not text_input.strip():
            st.warning("Please enter some ticket text first.")
        else:
            with st.spinner("Analyzing..."):
                result = process_ticket(
                    text_input.strip(), mt5_tok, mt5_model, marbert_tok, marbert_model,
                    use_api=use_api, api_provider=api_provider, api_model_name=api_model_name
                )
            # Stash in session_state so it survives reruns until the next click
            st.session_state["last_result"] = result

    # Always render from session_state (not a local var scoped to the button press) -
    # this, combined with giving each summary box a UNIQUE label below, is what fixes
    # the "output looks frozen on the next ticket" bug: the old code reused the exact
    # same widget label ("Summary") with a hardcoded static `key=`, so Streamlit kept
    # showing the first run's cached widget value instead of the new one.
    result = st.session_state.get("last_result")
    if result is not None:
        if result.get("needs_review"):
            st.warning("⚠️ Low-confidence output - flagged for human review.")

        st.subheader("🏆 Hybrid result (headline output)")
        c1, c2 = st.columns(2)
        c1.metric("Category", result.get("hybrid_category", "N/A"))
        sent = result.get("hybrid_sentiment", "N/A")
        c2.metric("Sentiment", f"{SENTIMENT_COLORS.get(sent, '')} {sent}")
        st.text_area("Hybrid Summary", result.get("hybrid_summary", ""), height=90, disabled=True)

        if result.get("used_api"):
            st.subheader(f"🌐 {result.get('api_provider_used')} API comparison ({result.get('api_model_used')})")
            if result.get("api_error"):
                st.error(f"API call failed: {result['api_error']}")
            else:
                d1, d2 = st.columns(2)
                d1.metric("Category", result["api_category"])
                asent = result["api_sentiment"]
                d2.metric("Sentiment", f"{SENTIMENT_COLORS.get(asent, '')} {asent}")
                st.text_area("Groq/OpenRouter Summary", result["api_summary"], height=90, disabled=True)

            with st.expander("Show raw individual model outputs (debugging)"):
                st.write("**MARBERTv2 only:**", {
                    "category": result.get("marbert_category"), "sentiment": result.get("marbert_sentiment")
                })
                st.write("**mT5 only:**", {
                    "category": result.get("mt5_category"), "sentiment": result.get("mt5_sentiment"),
                    "summary": result.get("mt5_summary"),
                })

# ── Batch tab ───────────────────────────────────────────────────────────
with tab_batch:
    st.write("Upload a CSV with a `text` column (one ticket per row).")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        if "text" not in batch_df.columns:
            st.error("CSV must have a column named `text`.")
        else:
            st.write(f"Loaded {len(batch_df)} tickets.")
            configured_ok_batch = True
            if use_api:
                configured_ok_batch = not PROVIDER_ENDPOINTS[api_provider]["api_key"].startswith("PASTE_YOUR_")
            can_run_batch = models_ready and (not use_api or configured_ok_batch)
            if st.button("Run batch analysis", type="primary", disabled=not can_run_batch):
                progress = st.progress(0.0)
                rows = []
                for i, row in batch_df.iterrows():
                    result = process_ticket(
                        str(row["text"]), mt5_tok, mt5_model, marbert_tok, marbert_model,
                        use_api=use_api, api_provider=api_provider, api_model_name=api_model_name
                    )
                    rows.append(result)
                    progress.progress((i + 1) / len(batch_df))

                results_df = pd.DataFrame(rows)
                display_cols = ["text", "hybrid_category", "hybrid_sentiment", "hybrid_summary", "needs_review"]
                if use_api:
                    display_cols += ["api_category", "api_sentiment", "api_summary"]
                st.dataframe(results_df[[c for c in display_cols if c in results_df.columns]], use_container_width=True)

                n_review = results_df["needs_review"].sum() if "needs_review" in results_df else 0
                st.caption(f"{n_review} ticket(s) flagged for human review (low-confidence output).")

                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    "Download full results as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="ticket_bot_results.csv",
                    mime="text/csv",
                )
