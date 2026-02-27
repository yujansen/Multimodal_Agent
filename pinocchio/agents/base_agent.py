"""BaseAgent — abstract foundation for all Pinocchio sub-agents.

Every cognitive-loop agent and multimodal processor inherits from this class.
It provides shared access to the LLM client, memory manager, and logger, as
well as a uniform ``run()`` interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pinocchio.memory.memory_manager import MemoryManager
from pinocchio.models.enums import AgentRole
from pinocchio.utils.llm_client import LLMClient
from pinocchio.utils.logger import PinocchioLogger


class BaseAgent(ABC):
    """Abstract base class for all Pinocchio sub-agents.

    Shared Capabilities:
      - Access to the LLM via ``self.llm``
      - Access to all three memory stores via ``self.memory``
      - Structured logging via ``self.logger``
      - A uniform ``run(**kwargs) -> Any`` interface
    """

    role: AgentRole = AgentRole.ORCHESTRATOR  # override in subclass

    def __init__(
        self,
        llm: LLMClient,
        memory: MemoryManager,
        logger: PinocchioLogger,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.logger = logger

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Execute this agent's primary function.

        Subclasses must override this method.
        """
        ...

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        self.logger.info(self.role, msg)

    def _warn(self, msg: str) -> None:
        self.logger.warn(self.role, msg)

    def _error(self, msg: str) -> None:
        self.logger.error(self.role, msg)
