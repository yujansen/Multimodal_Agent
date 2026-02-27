"""Pinocchio Logger — structured logging for agent activity.

Provides a consistent, colour-coded logging façade that tags every message
with the originating agent role for easy tracing.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from pinocchio.models.enums import AgentRole

# Colour codes (ANSI)
_COLOURS = {
    AgentRole.ORCHESTRATOR: "\033[1;36m",      # Bold Cyan
    AgentRole.PERCEPTION: "\033[0;33m",         # Yellow
    AgentRole.STRATEGY: "\033[0;35m",           # Magenta
    AgentRole.EXECUTION: "\033[0;32m",          # Green
    AgentRole.EVALUATION: "\033[0;34m",         # Blue
    AgentRole.LEARNING: "\033[1;33m",           # Bold Yellow
    AgentRole.META_REFLECTION: "\033[1;35m",    # Bold Magenta
    AgentRole.TEXT_PROCESSOR: "\033[0;37m",      # White
    AgentRole.VISION_PROCESSOR: "\033[0;31m",   # Red
    AgentRole.AUDIO_PROCESSOR: "\033[0;36m",    # Cyan
    AgentRole.VIDEO_PROCESSOR: "\033[1;31m",    # Bold Red
}
_RESET = "\033[0m"


class PinocchioLogger:
    """Structured logger for the Pinocchio agent system.

    Skills / Capabilities:
      - Role-tagged, colour-coded console output
      - Structured data logging (JSON-friendly dicts)
      - Configurable verbosity levels
      - Phase-transition markers for the cognitive loop
    """

    def __init__(self, level: int = logging.INFO) -> None:
        self._logger = logging.getLogger("pinocchio")
        self._logger.setLevel(level)
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)

    # ------------------------------------------------------------------
    # Core logging
    # ------------------------------------------------------------------

    def log(self, role: AgentRole, message: str, data: dict[str, Any] | None = None) -> None:
        colour = _COLOURS.get(role, "")
        prefix = f"{colour}[{role.value.upper()}]{_RESET}"
        line = f"{prefix} {message}"
        if data:
            import json
            line += f"\n  {json.dumps(data, ensure_ascii=False, indent=2)}"
        self._logger.info(line)

    def phase(self, phase_name: str) -> None:
        """Log a cognitive-loop phase transition."""
        bar = "─" * 50
        self._logger.info(f"\033[1;37m{bar}\n  ◆ {phase_name}\n{bar}{_RESET}")

    def separator(self) -> None:
        self._logger.info("\033[0;90m" + "═" * 60 + _RESET)

    # ------------------------------------------------------------------
    # Convenience shortcuts
    # ------------------------------------------------------------------

    def info(self, role: AgentRole, msg: str) -> None:
        self.log(role, msg)

    def warn(self, role: AgentRole, msg: str) -> None:
        self.log(role, f"⚠  {msg}")

    def error(self, role: AgentRole, msg: str) -> None:
        self.log(role, f"✖  {msg}")

    def success(self, role: AgentRole, msg: str) -> None:
        self.log(role, f"✔  {msg}")
