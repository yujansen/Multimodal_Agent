"""Pinocchio Orchestrator — the conductor of the cognitive loop.

This is the top-level agent that coordinates all sub-agents through the
complete self-evolving cognitive cycle:

    PERCEIVE → STRATEGIZE → EXECUTE → EVALUATE → LEARN → (META-REFLECT)

It also manages the multimodal processor pool, user model, and provides
the external API that callers interact with.

Skills / Capabilities (as the Orchestrator)
───────────────────────────────────────────
1. **Full Cognitive Loop Coordination**
   Drive the 6-phase self-learning cycle for every user interaction,
   ensuring no phase is skipped.

2. **Sub-Agent Dispatch**
   Route work to the appropriate sub-agent at each phase and pass
   context between them fluently.

3. **Multimodal Router**
   Automatically detect which modality processors are needed and
   invoke them via the Execution phase.

4. **User Model Management**
   Maintain and update the adaptive ``UserModel`` across interactions,
   tracking expertise level, communication style, and interests.

5. **Session Management**
   Track interaction count, manage conversation history, and provide
   a clean external API (``chat()``, ``reset()``).

6. **Meta-Reflection Scheduling**
   Trigger the MetaReflectionAgent at the configured interval.

7. **Error Recovery**
   Catch and recover from failures in any sub-agent, logging the error
   and falling back to a safe response.

8. **Memory Persistence**
   Ensure all memory stores are saved after each interaction.
"""

from __future__ import annotations

import traceback
from typing import Any

from pinocchio.agents.perception_agent import PerceptionAgent
from pinocchio.agents.strategy_agent import StrategyAgent
from pinocchio.agents.execution_agent import ExecutionAgent
from pinocchio.agents.evaluation_agent import EvaluationAgent
from pinocchio.agents.learning_agent import LearningAgent
from pinocchio.agents.meta_reflection_agent import MetaReflectionAgent
from pinocchio.memory.memory_manager import MemoryManager
from pinocchio.models.enums import AgentRole
from pinocchio.models.schemas import (
    AgentMessage,
    MultimodalInput,
    UserModel,
)
from pinocchio.multimodal.text_processor import TextProcessor
from pinocchio.multimodal.vision_processor import VisionProcessor
from pinocchio.multimodal.audio_processor import AudioProcessor
from pinocchio.multimodal.video_processor import VideoProcessor
from pinocchio.utils.llm_client import LLMClient
from pinocchio.utils.logger import PinocchioLogger


class Pinocchio:
    """Top-level orchestrator for the self-evolving multimodal agent.

    Usage
    -----
    >>> agent = Pinocchio()
    >>> response = agent.chat("Explain quantum entanglement simply.")
    >>> print(response)
    """

    GREETING = (
        "你好，我是 Pinocchio — 一个持续进化的多模态智能体。\n"
        "每一次对话都让我变得更好。我会记住有效的策略、从错误中学习、"
        "并不断精进我的推理能力。让我们开始吧。"
    )

    def __init__(
        self,
        model: str = "qwen2.5-omni",
        api_key: str | None = None,
        base_url: str | None = None,
        data_dir: str = "data",
        verbose: bool = True,
    ) -> None:
        # Shared infrastructure
        self.llm = LLMClient(model=model, api_key=api_key, base_url=base_url)
        self.memory = MemoryManager(data_dir=data_dir)
        self.logger = PinocchioLogger()

        # Cognitive-loop sub-agents
        self.perception = PerceptionAgent(self.llm, self.memory, self.logger)
        self.strategy = StrategyAgent(self.llm, self.memory, self.logger)
        self.execution = ExecutionAgent(self.llm, self.memory, self.logger)
        self.evaluation = EvaluationAgent(self.llm, self.memory, self.logger)
        self.learning = LearningAgent(self.llm, self.memory, self.logger)
        self.meta_reflection = MetaReflectionAgent(self.llm, self.memory, self.logger)

        # Multimodal processor pool
        self.text_proc = TextProcessor(self.llm, self.memory, self.logger)
        self.vision_proc = VisionProcessor(self.llm, self.memory, self.logger)
        self.audio_proc = AudioProcessor(self.llm, self.memory, self.logger)
        self.video_proc = VideoProcessor(self.llm, self.memory, self.logger)

        # Session state
        self.user_model = UserModel()
        self.conversation_history: list[dict[str, str]] = []
        self._interaction_count = 0
        self._verbose = verbose

    # ------------------------------------------------------------------
    # External API
    # ------------------------------------------------------------------

    def chat(
        self,
        text: str | None = None,
        *,
        image_paths: list[str] | None = None,
        audio_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
    ) -> str:
        """Process a user message through the full cognitive loop.

        Parameters
        ----------
        text : User's text message (may be None for pure media input).
        image_paths : Optional list of image file paths or URLs.
        audio_paths : Optional list of audio file paths.
        video_paths : Optional list of video file paths.

        Returns
        -------
        str : The agent's response text.
        """
        self._interaction_count += 1
        self.user_model.interaction_count = self._interaction_count
        self.logger.separator()
        self.logger.log(
            AgentRole.ORCHESTRATOR,
            f"Interaction #{self._interaction_count}",
        )

        user_input = MultimodalInput(
            text=text,
            image_paths=image_paths or [],
            audio_paths=audio_paths or [],
            video_paths=video_paths or [],
        )

        try:
            response = self._run_cognitive_loop(user_input)
        except Exception as exc:
            self.logger.error(AgentRole.ORCHESTRATOR, f"Cognitive loop error: {exc}")
            traceback.print_exc()
            response = AgentMessage(
                content="抱歉，处理过程中出现了意外错误。请重新尝试您的请求。",
                confidence=0.0,
            )

        # Store in conversation history
        if text:
            self.conversation_history.append({"role": "user", "content": text})
        self.conversation_history.append(
            {"role": "assistant", "content": response.content}
        )

        return response.content

    def greet(self) -> str:
        """Return the initialization greeting."""
        return self.GREETING

    def reset(self) -> None:
        """Reset session state (keeps persistent memory)."""
        self.conversation_history.clear()
        self._interaction_count = 0
        self.user_model = UserModel()

    def status(self) -> dict[str, Any]:
        """Return a summary of the agent's current state."""
        return {
            "interaction_count": self._interaction_count,
            "memory_summary": self.memory.summary(),
            "improvement_trend": self.memory.improvement_trend(),
            "user_model": {
                "expertise": self.user_model.expertise.value,
                "style": self.user_model.style.value,
                "interests": self.user_model.domains_of_interest,
            },
        }

    # ------------------------------------------------------------------
    # Internal: Cognitive Loop
    # ------------------------------------------------------------------

    def _run_cognitive_loop(self, user_input: MultimodalInput) -> AgentMessage:
        """Execute the full PERCEIVE → STRATEGIZE → EXECUTE → EVALUATE → LEARN pipeline."""

        # Phase 1: PERCEIVE
        perception = self.perception.run(user_input=user_input)

        # Phase 2: STRATEGIZE
        strategy = self.strategy.run(perception=perception)

        # Phase 3: EXECUTE
        response = self.execution.run(
            user_input=user_input,
            perception=perception,
            strategy=strategy,
        )

        # Phase 4: EVALUATE
        evaluation = self.evaluation.run(
            user_input=user_input,
            perception=perception,
            strategy=strategy,
            response=response,
        )

        # Phase 5: LEARN
        user_text = user_input.text or "(non-text input)"
        learning = self.learning.run(
            user_input_text=user_text,
            perception=perception,
            strategy=strategy,
            evaluation=evaluation,
        )

        # Phase 6: META-REFLECT (periodic)
        if self.meta_reflection.should_trigger():
            self.logger.log(
                AgentRole.ORCHESTRATOR,
                "Meta-reflection triggered — running higher-order analysis",
            )
            meta = self.meta_reflection.run()
            # Log key insights
            if meta.priority_improvements:
                self.logger.log(
                    AgentRole.ORCHESTRATOR,
                    f"Top improvement priorities: {meta.priority_improvements[:3]}",
                )

        self.logger.separator()
        self.logger.log(
            AgentRole.ORCHESTRATOR,
            f"Interaction #{self._interaction_count} complete — "
            f"quality: {evaluation.output_quality}/10",
        )

        return response
