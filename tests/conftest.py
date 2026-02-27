"""Test configuration and shared fixtures."""

from __future__ import annotations

import json
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Provide a temporary directory for memory persistence tests."""
    return str(tmp_path)


@pytest.fixture
def mock_llm():
    """Return a MagicMock that mimics LLMClient."""
    llm = MagicMock()
    llm.model = "test-model"
    llm.temperature = 0.7
    llm.max_tokens = 4096
    return llm


@pytest.fixture
def mock_llm_with_json(mock_llm):
    """LLM mock that returns configurable JSON responses."""
    def _setup(json_response: dict):
        mock_llm.ask_json.return_value = json_response
        mock_llm.ask.return_value = "Test response from LLM"
        mock_llm.chat.return_value = "Test chat response from LLM"
        return mock_llm
    return _setup
