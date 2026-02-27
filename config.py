"""Pinocchio agent configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class PinocchioConfig:
    """Configuration for the Pinocchio agent.

    All values can be overridden via environment variables prefixed with
    ``PINOCCHIO_``.
    """

    # LLM settings (Qwen2.5-Omni via Ollama local server)
    model: str = field(default_factory=lambda: os.getenv("PINOCCHIO_MODEL", "qwen2.5-omni"))
    api_key: str = field(default_factory=lambda: os.getenv("OLLAMA_API_KEY", "ollama"))
    base_url: str | None = field(default_factory=lambda: os.getenv(
        "OPENAI_BASE_URL", "http://localhost:11434/v1",
    ))
    temperature: float = 0.7
    max_tokens: int = 4096

    # Memory settings
    data_dir: str = field(default_factory=lambda: os.getenv("PINOCCHIO_DATA_DIR", "data"))

    # Behaviour settings
    meta_reflect_interval: int = 5
    verbose: bool = True
