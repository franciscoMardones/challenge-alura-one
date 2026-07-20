"""Streamlit frontend for BimBam Buy RAG Agent.

Provides a branded, streaming chat interface that communicates with the FastAPI
backend via SSE (Server-Sent Events).
"""

import os

import streamlit as st
import httpx
import json
import uuid
import re
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL = os.getenv("API_URL", "http://localhost:8000")
CHAT_STREAM_ENDPOINT = f"{API_URL}/chat/stream"
HEALTH_ENDPOINT = f"{API_URL}/health"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="BimBam Buy - Asistente Virtual",
    page_icon="🤖",
    layout="centered",
)

# ---------------------------------------------------------------------------
# CSS injection — brand colors, bubble styling, typography
# ---------------------------------------------------------------------------

BRAND_CSS = """
<style>
/* ---- Brand colors ---- */
:root {
    --bimbam-primary: #FF6B35;
    --bimbam-secondary: #004E89;
}

/* ---- Header ---- */
.stApp header {
    border-bottom: 3px solid var(--bimbam-primary);
}

/* ---- Chat bubble overrides ---- */
.stChatMessage {
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 8px;
    border-left: 4px solid transparent;
}
[data-testid="stChatMessage"][aria-label="user"] {
    border-left-color: var(--bimbam-secondary);
}
[data-testid="stChatMessage"][aria-label="assistant"] {
    border-left-color: var(--bimbam-primary);
}

/* ---- Timestamp caption ---- */
.msg-timestamp {
    font-size: 0.75em;
    color: #888;
    margin-top: 4px;
    text-align: right;
}

/* ---- Typography ---- */
.stMarkdown p { line-height: 1.6; }
.stMarkdown h2 { color: var(--bimbam-secondary); }

/* ---- Skeleton shimmer ---- */
@keyframes shimmer {
    0%   { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}
.skeleton-line {
    height: 16px;
    border-radius: 4px;
    background: linear-gradient(90deg, #e0e0e0 25%, #f0f0f0 50%, #e0e0e0 75%);
    background-size: 800px 100%;
    animation: shimmer 1.5s infinite linear;
    margin-bottom: 10px;
}
.skeleton-line-short  { width: 40%; }
.skeleton-line-long   { width: 90%; }
.skeleton-line-medium { width: 65%; }
</style>
"""

st.markdown(BRAND_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "welcome_shown" not in st.session_state:
    st.session_state.welcome_shown = True

if "welcome_question" not in st.session_state:
    st.session_state.welcome_question = None

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🤖 BimBam Buy - Asistente Virtual")
st.caption("Pregúntale sobre devoluciones, afiliados, métodos de pago y más")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STREAM_URL = f"{API_URL}/chat/stream"


def _detect_language(text: str) -> str:
    """Simple heuristic to detect if response is Spanish or English."""
    spanish_indicators = [
        r'\b(el|la|los|las|un|una|de|del|en|con|por|para|que|se|no|sí|pero|como|este|esta|más|también|puede|tiene|debe|todo|bien)\b',
        r'[áéíóúñ¿¡]',
    ]
    score = 0
    for pattern in spanish_indicators:
        if re.search(pattern, text, re.IGNORECASE):
            score += 1
    return "Español" if score >= 2 else "English"


def _now_timestamp() -> str:
    """Return current time as HH:MM."""
    return datetime.now().strftime("%H:%M")


def render_skeleton() -> str:
    """Return skeleton HTML to show animated shimmer placeholders."""
    return """
    <div class="skeleton-container">
      <div class="skeleton-line skeleton-line-short"></div>
      <div class="skeleton-line skeleton-line-long"></div>
      <div class="skeleton-line skeleton-line-medium"></div>
    </div>
    """


def render_welcome() -> None:
    """Render the branded welcome section with clickable topic cards."""
    st.markdown("### 👋 ¡Bienvenido a BimBam Buy!")
    st.markdown("**BimBam Buy** es tu tienda online de confianza.")
    st.markdown("Este chat puede ayudarte con:")

    # Create clickable cards using Streamlit columns + buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📋 Política de devoluciones", key="card_devoluciones", use_container_width=True):
            st.session_state.welcome_question = "¿Cuál es la política de devoluciones?"
    with col2:
        if st.button("💰 Métodos de pago", key="card_pagos", use_container_width=True):
            st.session_state.welcome_question = "¿Qué métodos de pago aceptan?"
    with col3:
        if st.button("🚚 Costos de envío", key="card_envios", use_container_width=True):
            st.session_state.welcome_question = "¿Cuánto cuesta el envío?"

    col4, col5 = st.columns(2)
    with col4:
        if st.button("🤝 Programa de afiliados", key="card_afiliados", use_container_width=True):
            st.session_state.welcome_question = "¿Cómo funciona el programa de afiliados?"
    with col5:
        if st.button("🛡️ Garantías de productos", key="card_garantias", use_container_width=True):
            st.session_state.welcome_question = "¿Cuál es la garantía de los productos?"

    st.markdown("*Ejemplo: \"¿Cuántos días tengo para devolver un producto?\"*")


def handle_chat(prompt: str) -> None:
    """Send a message to the streaming endpoint and render the response progressively."""
    # Append user message
    user_msg = {
        "role": "user",
        "content": prompt,
        "timestamp": _now_timestamp(),
    }
    st.session_state.messages.append(user_msg)

    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)
        st.caption(f"🕐 {user_msg['timestamp']}")

    # Create placeholder for assistant response
    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        placeholder.markdown(render_skeleton(), unsafe_allow_html=True)

        try:
            answer = ""
            sources = []

            with httpx.stream(
                "POST",
                STREAM_URL,
                json={
                    "message": prompt,
                    "session_id": st.session_state.session_id,
                },
                timeout=60.0,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = json.loads(line[6:])
                    if "token" in payload:
                        answer += payload["token"]
                        placeholder.markdown(answer)
                    if payload.get("done"):
                        sources = payload.get("sources", [])

            # Clear skeleton, show final answer
            placeholder.markdown(answer)

            # Timestamp
            st.caption(f"🕐 {_now_timestamp()}")

            # Language badge
            lang = _detect_language(answer)
            st.caption(f"🌐 Idioma detectado: {lang}")

            # Sources expander
            if sources:
                with st.expander("📄 Fuentes consultadas"):
                    for source in sources:
                        st.write(f"- {source}")

            # Store in history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "timestamp": _now_timestamp(),
                "error": False,
            })

        except (httpx.ConnectError, httpx.HTTPStatusError, Exception) as e:
            error_msg = f"⚠️ Error: {str(e)}"
            placeholder.empty()
            st.error(error_msg)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "sources": [],
                "timestamp": _now_timestamp(),
                "error": True,
            })
            st.session_state.last_failed_prompt = prompt


# ---------------------------------------------------------------------------
# Welcome section (dismissible after first message)
# ---------------------------------------------------------------------------

if st.session_state.welcome_shown and len(st.session_state.messages) == 0:
    render_welcome()

if len(st.session_state.messages) > 0:
    st.session_state.welcome_shown = False

# Handle welcome card clicks
if st.session_state.welcome_question:
    question = st.session_state.welcome_question
    st.session_state.welcome_question = None
    st.session_state.welcome_shown = False
    handle_chat(question)
    st.rerun()

# ---------------------------------------------------------------------------
# Display chat history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        # Timestamp
        if msg.get("timestamp"):
            st.caption(f"🕐 {msg['timestamp']}")
        # Sources (assistant only)
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Fuentes consultadas"):
                for source in msg["sources"]:
                    st.write(f"- {source}")
        # Language badge
        if msg["role"] == "assistant":
            lang = _detect_language(msg["content"])
            st.caption(f"🌐 Idioma detectado: {lang}")


# ---------------------------------------------------------------------------
# Retry button (on last error)
# ---------------------------------------------------------------------------

if st.session_state.get("last_failed_prompt"):
    if st.button("🔄 Reintentar"):
        failed = st.session_state.last_failed_prompt
        del st.session_state.last_failed_prompt
        handle_chat(failed)
        st.rerun()

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    handle_chat(prompt)

# ---------------------------------------------------------------------------
# Sidebar: session info
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("ℹ️ Información")
    st.write(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
    st.write(f"**Mensajes:** {len(st.session_state.messages)}")

    if st.button("🗑️ Nueva conversación"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.welcome_shown = True
        st.rerun()

    st.divider()
    st.markdown("""
    **Ejemplos de preguntas:**
    - ¿Cuál es la política de devoluciones?
    - ¿Cómo funciona el programa de afiliados?
    - ¿Qué métodos de pago aceptan?
    - ¿Cuánto cuesta el envío?
    - ¿Cuál es la garantía de los productos?
    """)
