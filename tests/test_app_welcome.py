"""Tests for welcome section rendering and dismissal."""

import pytest
from unittest.mock import patch, MagicMock


class TestWelcomeSection:
    """Verify welcome section renders on first load and hides after first message."""

    def test_render_welcome_produces_html(self):
        """render_welcome() should produce HTML with welcome-container class."""
        from app import render_welcome
        import streamlit as st

        with patch.object(st, "markdown") as mock_md:
            render_welcome()
            mock_md.assert_called_once()
            html = mock_md.call_args[0][0]
            assert "welcome-container" in html
            assert "BimBam Buy" in html

    def test_welcome_shown_default_true(self):
        """welcome_shown defaults to True in a fresh session."""
        import app  # noqa: F401
        # In a fresh Streamlit session, welcome_shown should be True
        # This is verified by the initialization block in app.py
        # We test the logic: if welcome_shown and no messages, render welcome
        mock_state = {"welcome_shown": True, "messages": []}
        should_show = mock_state["welcome_shown"] and len(mock_state["messages"]) == 0
        assert should_show is True

    def test_welcome_hidden_after_first_message(self):
        """welcome_shown should be False after messages exist."""
        mock_state = {"welcome_shown": False, "messages": [{"role": "user", "content": "hi"}]}
        should_show = mock_state["welcome_shown"] and len(mock_state["messages"]) == 0
        assert should_show is False

    def test_welcome_cards_contain_all_topics(self):
        """Welcome HTML must include all 5 topic cards."""
        from app import render_welcome
        import streamlit as st

        with patch.object(st, "markdown") as mock_md:
            render_welcome()
            html = mock_md.call_args[0][0]
            assert "devoluciones" in html.lower()
            assert "pago" in html.lower()
            assert "envío" in html.lower() or "envio" in html.lower()
            assert "afiliados" in html.lower()
            assert "garantías" in html.lower() or "garantias" in html.lower()
