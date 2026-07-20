"""FastAPI backend for BimBam Buy RAG Agent.

Provides POST /chat and GET /health endpoints with session memory,
CORS, and error handling.
"""

import os
import re
import time
import json
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import CHROMA_DIR, EMBEDDING_MODEL
from rag import run_rag, run_rag_stream

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SESSION_TTL_SECONDS = 30 * 60  # 30 minutes
MAX_SESSION_MESSAGES = 10      # 5 user + 5 assistant
MAX_MESSAGE_LENGTH = 2000      # Max characters per user message
SESSION_ID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")

# CORS — restrict to known origins (comma-separated env var)
ALLOWED_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")
    if origin.strip()
]

# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------

sessions: dict[str, list[dict]] = {}
session_timestamps: dict[str, float] = {}


def _cleanup_expired_sessions() -> None:
    """Remove sessions inactive for more than 30 minutes."""
    now = time.time()
    expired = [
        sid for sid, ts in session_timestamps.items()
        if now - ts > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        sessions.pop(sid, None)
        session_timestamps.pop(sid, None)


def _validate_session_id(session_id: str) -> bool:
    """Validate that session_id is a proper UUID v4."""
    return bool(SESSION_ID_PATTERN.match(session_id))


def _get_session_history(session_id: str) -> list[dict]:
    """Get or create session history, cleaning up expired sessions first."""
    _cleanup_expired_sessions()
    if session_id not in sessions:
        sessions[session_id] = []
    session_timestamps[session_id] = time.time()
    return sessions[session_id]


def _add_message(session_id: str, role: str, content: str) -> None:
    """Append a message, keeping only the last MAX_SESSION_MESSAGES."""
    history = sessions[session_id]
    history.append({"role": role, "content": content})
    # Keep only last N messages
    if len(history) > MAX_SESSION_MESSAGES:
        sessions[session_id] = history[-MAX_SESSION_MESSAGES:]


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    session_id: str = Field(..., min_length=36, max_length=36)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    session_id: str


class HealthResponse(BaseModel):
    status: str
    documents_loaded: bool


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------

vectorstore: Optional[Chroma] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load vector store at startup, clean up at shutdown."""
    global vectorstore
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    print(f"Vector store loaded: {vectorstore._collection.count()} vectors")
    yield
    # Shutdown cleanup if needed
    vectorstore = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="BimBam Buy RAG Agent", version="0.3.0", lifespan=lifespan)

# CORS — restrict to known origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Process a chat message using RAG."""
    global vectorstore

    if vectorstore is None:
        raise HTTPException(status_code=503, detail="Vector store not ready")

    if not _validate_session_id(req.session_id):
        raise HTTPException(status_code=422, detail="Invalid session ID format")

    try:
        # Build context with conversation history
        history = _get_session_history(req.session_id)
        history_text = ""
        if history:
            history_text = "\n".join(
                f"{'Usuario' if m['role'] == 'user' else 'Asistente'}: {m['content']}"
                for m in history[-6:]  # Last 3 exchanges
            )
            history_text = f"\n\nHistorial reciente:\n{history_text}"

        # Run RAG pipeline (includes retrieval, fallback check, generation)
        result = run_rag(req.message, vectorstore, history_text=history_text)

        # Store messages in session
        _add_message(req.session_id, "user", req.message)
        _add_message(req.session_id, "assistant", result["answer"])

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=req.session_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Stream RAG response as SSE events.

    Each event is a JSON object on a `data:` line:
        {"token": "..."}                    — streamed token
        {"sources": [...], "done": true}    — final event
    """
    global vectorstore

    if vectorstore is None:
        raise HTTPException(status_code=503, detail="Vector store not ready")

    if not _validate_session_id(req.session_id):
        raise HTTPException(status_code=422, detail="Invalid session ID format")

    async def event_generator() -> AsyncGenerator[str, None]:
        # Build conversation history context
        history = _get_session_history(req.session_id)
        history_text = ""
        if history:
            history_text = "\n".join(
                f"{'Usuario' if m['role'] == 'user' else 'Asistente'}: {m['content']}"
                for m in history[-6:]
            )
            history_text = f"\n\nHistorial reciente:\n{history_text}"

        full_answer = ""
        for event in run_rag_stream(req.message, vectorstore, history_text=history_text):
            if "token" in event:
                full_answer += event["token"]
            yield f"data: {json.dumps(event)}\n\n"

        # Store in session after stream completes
        _add_message(req.session_id, "user", req.message)
        _add_message(req.session_id, "assistant", full_answer)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    docs_loaded = vectorstore is not None and vectorstore._collection.count() > 0
    return HealthResponse(status="ok", documents_loaded=docs_loaded)
