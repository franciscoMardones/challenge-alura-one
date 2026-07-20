"""Tests for api.py /chat/stream SSE endpoint."""

import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _parse_sse_events(text: str) -> list[dict]:
    """Parse SSE text into list of JSON dicts."""
    events = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            payload = line[len("data: "):]
            if payload == "[DONE]":
                continue
            events.append(json.loads(payload))
    return events


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

@patch("api.vectorstore")
def test_chat_stream_returns_sse_content_type(mock_vs):
    """POST /chat/stream must return text/event-stream content type."""
    from api import app

    mock_vs._collection.count.return_value = 10

    client = TestClient(app)
    resp = client.post(
        "/chat/stream",
        json={"message": "test", "session_id": "s1"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]


@patch("api.vectorstore")
def test_chat_stream_emits_token_events(mock_vs):
    """SSE response must contain token events with {'token': ...} shape."""
    from api import app

    mock_vs._collection.count.return_value = 10

    def fake_stream_gen(q, vs, history_text=""):
        yield {"token": "Hola"}
        yield {"token": " "}
        yield {"token": "mundo"}
        yield {"sources": ["faq.pdf"], "done": True}

    with patch("api.run_rag_stream", side_effect=fake_stream_gen):
        client = TestClient(app)
        resp = client.post(
            "/chat/stream",
            json={"message": "test", "session_id": "s2"},
        )

    events = _parse_sse_events(resp.text)
    token_events = [e for e in events if "token" in e]
    assert len(token_events) == 3
    assert token_events[0] == {"token": "Hola"}


@patch("api.vectorstore")
def test_chat_stream_emits_done_event(mock_vs):
    """SSE response must end with a done event containing sources."""
    from api import app

    mock_vs._collection.count.return_value = 10

    def fake_stream_gen(q, vs, history_text=""):
        yield {"token": "ok"}
        yield {"sources": ["a.pdf", "b.pdf"], "done": True}

    with patch("api.run_rag_stream", side_effect=fake_stream_gen):
        client = TestClient(app)
        resp = client.post(
            "/chat/stream",
            json={"message": "hi", "session_id": "s3"},
        )

    events = _parse_sse_events(resp.text)
    done_events = [e for e in events if e.get("done") is True]
    assert len(done_events) == 1
    assert done_events[0]["sources"] == ["a.pdf", "b.pdf"]


@patch("api.vectorstore")
def test_chat_stream_empty_message_returns_422(mock_vs):
    """Empty message must be rejected with 422."""
    from api import app

    client = TestClient(app)
    resp = client.post(
        "/chat/stream",
        json={"message": "", "session_id": "s4"},
    )
    assert resp.status_code == 422


@patch("api.vectorstore")
def test_chat_stream_no_vectorstore_returns_503(mock_vs):
    """If vectorstore is None, return 503."""
    from api import app

    import api as api_module
    original = api_module.vectorstore
    api_module.vectorstore = None
    try:
        client = TestClient(app)
        resp = client.post(
            "/chat/stream",
            json={"message": "test", "session_id": "s5"},
        )
        assert resp.status_code == 503
    finally:
        api_module.vectorstore = original
