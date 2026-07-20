"""Tests for rag.run_rag_stream — streaming RAG generator."""

import pytest
from unittest.mock import MagicMock, patch


class FakeChunk:
    """Minimal object mimicking a LangChain StrOutputParser chunk."""

    def __init__(self, text: str):
        self.text = text

    def __str__(self):
        return self.text


def _fake_stream_tokens(*tokens: str):
    """Yield FakeChunk objects like chain.stream()."""
    for t in tokens:
        yield FakeChunk(t)


# -----------------------------------------------------------------------
# run_rag_stream yields token dicts with {"token": ...} shape
# -----------------------------------------------------------------------

@patch("rag.build_rag_chain")
@patch("rag.get_relevant_docs")
def test_run_rag_stream_yields_token_dicts(mock_docs, mock_chain):
    """Each streamed chunk must be a dict with a 'token' key."""
    from rag import run_rag_stream

    doc = MagicMock()
    doc.metadata = {"source": "faq.pdf"}
    doc.page_content = "content"
    mock_docs.return_value = ([doc], ["faq.pdf"])

    fake_chain = MagicMock()
    fake_chain.stream.return_value = _fake_stream_tokens("Hola", " Mundo")
    mock_chain.return_value = fake_chain

    tokens = list(run_rag_stream("test", MagicMock(), history_text=""))

    # Filter out the final "done" event
    token_events = [t for t in tokens if "token" in t]
    assert len(token_events) == 2
    assert token_events[0] == {"token": "Hola"}
    assert token_events[1] == {"token": " Mundo"}


@patch("rag.build_rag_chain")
@patch("rag.get_relevant_docs")
def test_run_rag_stream_yields_done_event(mock_docs, mock_chain):
    """The last yielded event must contain 'sources' and 'done': True."""
    from rag import run_rag_stream

    doc = MagicMock()
    doc.metadata = {"source": "pago.pdf"}
    doc.page_content = "pago info"
    mock_docs.return_value = ([doc], ["pago.pdf"])

    fake_chain = MagicMock()
    fake_chain.stream.return_value = _fake_stream_tokens("respuesta")
    mock_chain.return_value = fake_chain

    events = list(run_rag_stream("test", MagicMock(), history_text=""))
    done_events = [e for e in events if e.get("done") is True]

    assert len(done_events) == 1
    assert done_events[0]["sources"] == ["pago.pdf"]
    assert done_events[0]["done"] is True


@patch("rag.get_relevant_docs")
def test_run_rag_stream_empty_docs_yields_error(mock_docs):
    """When no docs found, should yield an error event with fallback message."""
    from rag import run_rag_stream

    mock_docs.return_value = ([], [])

    events = list(run_rag_stream("test", MagicMock(), history_text=""))
    done_events = [e for e in events if e.get("done") is True]

    assert len(done_events) == 1
    assert done_events[0]["sources"] == []
    assert done_events[0]["done"] is True


@patch("rag.build_rag_chain")
@patch("rag.get_relevant_docs")
def test_run_rag_stream_includes_history(mock_docs, mock_chain):
    """History text must be appended to context before chain invocation."""
    from rag import run_rag_stream

    doc = MagicMock()
    doc.metadata = {"source": "x.pdf"}
    doc.page_content = "x"
    mock_docs.return_value = ([doc], ["x.pdf"])

    fake_chain = MagicMock()
    fake_chain.stream.return_value = _fake_stream_tokens("ok")
    mock_chain.return_value = fake_chain

    list(run_rag_stream("q", MagicMock(), history_text="\n\nHistorial: User: hola"))

    call_args = fake_chain.stream.call_args[0][0]
    assert "Historial: User: hola" in call_args["context"]
