"""
tests/test_config_and_startup.py

Tests for configuration management, graceful degradation when GROQ_API_KEY is missing,
and copilot import stability.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import app_config


class TestAppConfig:
    """Verifies that app_config behaves safely under all environment states."""

    def test_missing_api_key_does_not_crash(self, monkeypatch):
        """When GROQ_API_KEY is not in env or secrets, should return empty string without raising."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with patch.object(app_config.os, "getenv", return_value=""):
            key = app_config.get_groq_api_key()
            assert key == ""
            assert app_config.is_groq_available() is False

    def test_valid_env_var_retrieval(self, monkeypatch):
        """When GROQ_API_KEY is set in environment, should retrieve it correctly."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_valid_test_key_12345")
        key = app_config.get_groq_api_key()
        assert key == "gsk_valid_test_key_12345"
        assert app_config.is_groq_available() is True

    def test_streamlit_secrets_fallback(self, monkeypatch):
        """When env var is absent but st.secrets exists, should use secrets safely."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        mock_st = MagicMock()
        mock_st.secrets = {"GROQ_API_KEY": "gsk_secret_key_67890"}

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            with patch.object(app_config.os, "getenv", return_value=""):
                key = app_config.get_groq_api_key()
                assert key == "gsk_secret_key_67890"
                assert app_config.is_groq_available() is True

    def test_streamlit_secrets_permission_or_missing_handled_safely(self, monkeypatch):
        """When st.secrets raises an exception, should safely catch it and return empty."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        mock_st = MagicMock()
        # Simulate StreamlitSecretNotFoundError or PermissionError
        mock_st.secrets.get.side_effect = PermissionError("Access denied to secrets.toml")

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            with patch.object(app_config.os, "getenv", return_value=""):
                key = app_config.get_groq_api_key()
                assert key == ""
                assert app_config.is_groq_available() is False


class TestCopilotImportAndAsk:
    """Verifies that agents.copilot.ask is cleanly importable and functions under all conditions."""

    def test_ask_is_importable_and_callable(self):
        """Verifies requirement: ask() must be defined and callable."""
        from agents.copilot import ask
        assert callable(ask)

    def test_ask_deterministic_sql_query(self, monkeypatch):
        """Questions routed to SQL work without requiring external LLM API calls."""
        from agents.copilot import ask
        res = ask("How many incidents are there?")
        assert isinstance(res, str)
        assert len(res) > 0

    def test_ask_deterministic_sop_query(self):
        """Questions routed to safety SOP work without requiring external LLM API calls."""
        from agents.copilot import ask
        res = ask("What is the helmet safety procedure?")
        assert isinstance(res, str)
        assert len(res) > 0

    def test_ask_deterministic_stats_query(self):
        """Questions routed to dashboard stats work without requiring external LLM API calls."""
        from agents.copilot import ask
        res = ask("Show factory dashboard")
        assert isinstance(res, str)
        assert "Factory Dashboard" in res or "Total Incidents" in res

    def test_ask_graceful_message_when_key_missing_for_llm(self, monkeypatch):
        """When query falls back to LLM agent but key is missing, returns clean message."""
        from agents.copilot import ask
        with patch("app_config.is_groq_available", return_value=False):
            res = ask("What is the general philosophy of industrial safety?")
            assert "GROQ_API_KEY is not configured" in res
