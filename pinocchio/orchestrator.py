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

import asyncio
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from pinocchio.utils.resource_monitor import ResourceMonitor


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
        max_workers: int | None = None,
        parallel_modalities: bool = True,
        meta_reflect_interval: int = 5,
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
        self.meta_reflection = MetaReflectionAgent(
            self.llm, self.memory, self.logger,
            meta_reflect_interval=meta_reflect_interval,
        )

        # Multimodal processor pool
        self.text_proc = TextProcessor(self.llm, self.memory, self.logger)
        self.vision_proc = VisionProcessor(self.llm, self.memory, self.logger)
        self.audio_proc = AudioProcessor(self.llm, self.memory, self.logger)
        self.video_proc = VideoProcessor(self.llm, self.memory, self.logger)

        # Resource detection & parallelism
        self._resource_monitor = ResourceMonitor()
        self._resources = self._resource_monitor.snapshot()
        self._max_workers = max_workers or self._resources.recommended_workers
        self._parallel_modalities = parallel_modalities

        # Session state
        self.user_model = UserModel()
        self.conversation_history: list[dict[str, str]] = []
        self._interaction_count = 0
        self._verbose = verbose
        self._lock = threading.Lock()  # protects session state mutations

        # Log detected hardware
        if verbose:
            r = self._resources
            gpu_label = (
                f"{r.gpus[0].name} ({r.total_vram_mb:,} MB VRAM)"
                if r.has_gpu else "none"
            )
            self.logger.log(
                AgentRole.ORCHESTRATOR,
                f"Hardware: {r.cpu_count_physical} cores, "
                f"{r.ram_total_mb:,} MB RAM, GPU: {gpu_label} | "
                f"Workers: {self._max_workers}, "
                f"Parallel modalities: {self._parallel_modalities}",
            )

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
        with self._lock:
            self._interaction_count += 1
            interaction_num = self._interaction_count
            self.user_model.interaction_count = interaction_num
        self.logger.separator()
        self.logger.log(
            AgentRole.ORCHESTRATOR,
            f"Interaction #{interaction_num}",
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
        with self._lock:
            if text:
                self.conversation_history.append({"role": "user", "content": text})
            self.conversation_history.append(
                {"role": "assistant", "content": response.content}
            )

        return response.content

    async def async_chat(
        self,
        text: str | None = None,
        *,
        image_paths: list[str] | None = None,
        audio_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
    ) -> str:
        """Async version of :meth:`chat` — runs the cognitive loop off the event loop.

        This wraps the synchronous cognitive loop in ``asyncio.to_thread``
        so it doesn't block other coroutines.  For truly async modality
        preprocessing, use :class:`AsyncLLMClient`.

        Usage
        -----
        >>> response = await agent.async_chat("Explain quantum entanglement.")
        """
        return await asyncio.to_thread(
            self.chat,
            text,
            image_paths=image_paths,
            audio_paths=audio_paths,
            video_paths=video_paths,
        )

    def greet(self) -> str:
        """Return the initialization greeting."""
        return self.GREETING

    def reset(self) -> None:
        """Reset session state (keeps persistent memory)."""
        self.conversation_history.clear()
        self._interaction_count = 0
        self.user_model = UserModel()
        self.memory.reset_working_memory()

    def status(self) -> dict[str, Any]:
        """Return a summary of the agent's current state."""
        self._resources = self._resource_monitor.snapshot(refresh=True)
        return {
            "interaction_count": self._interaction_count,
            "memory_summary": self.memory.summary(),
            "improvement_trend": self.memory.improvement_trend(),
            "user_model": {
                "expertise": self.user_model.expertise.value,
                "style": self.user_model.style.value,
                "interests": self.user_model.domains_of_interest,
            },
            "resources": self._resources.to_dict(),
            "working_memory": self.memory.working.summary(),
        }

    # ------------------------------------------------------------------
    # Internal: Cognitive Loop
    # ------------------------------------------------------------------

    MAX_COMPLETION_RETRIES = 3  # max continuation attempts for incomplete responses

    def _run_cognitive_loop(self, user_input: MultimodalInput) -> AgentMessage:
        """Execute the full PERCEIVE → STRATEGIZE → EXECUTE → EVALUATE → LEARN pipeline.

        This is the heart of Pinocchio's self-evolving intelligence.  Every
        user interaction passes through all 6 phases, each handled by a
        dedicated sub-agent.  Phase 4.5 (retry loop) ensures response
        completeness, and Phase 6 meta-reflect fires periodically for
        higher-order self-improvement.
        """

        # ── Phase 0: Buffer the raw user input into working memory ──
        # Working memory gives downstream agents conversational context
        if user_input.text:
            self.memory.working.add_conversation_turn("user", user_input.text)

        # ── Phase 0.5: PREPROCESS MODALITIES (parallel or sequential) ──
        # Non-text modalities (images/audio/video) are converted to text
        # descriptions before entering the cognitive loop so every agent
        # can reason about them uniformly.
        modality_context = self._preprocess_modalities(user_input)

        # ── Phase 1: PERCEIVE ──
        # Classify the task, detect modalities, assess complexity & confidence
        perception = self.perception.run(
            user_input=user_input,
            modality_context=modality_context,
        )

        # ── Phase 2: STRATEGIZE ──
        # Select (or construct) the best approach; retrieve procedural memory
        strategy = self.strategy.run(perception=perception)

        # ── Phase 3: EXECUTE ──
        # Generate the user-facing response; includes auto-continuation if
        # the LLM hits its token limit (up to _MAX_AUTO_CONTINUATIONS rounds)
        response = self.execution.run(
            user_input=user_input,
            perception=perception,
            strategy=strategy,
            modality_context=modality_context,
        )

        # ── Phase 4: EVALUATE ──
        # Score quality, check completeness, identify improvement areas
        evaluation = self.evaluation.run(
            user_input=user_input,
            perception=perception,
            strategy=strategy,
            response=response,
        )

        # ── Phase 4.5: COMPLETENESS RETRY LOOP ──
        # If the evaluator says the response is incomplete, ask the
        # execution agent to continue.  This is the *outer* retry loop;
        # each continue_response() call also has its own *inner* auto-
        # continuation loop for token-limit hits.
        retry_count = 0
        while not evaluation.is_complete and retry_count < self.MAX_COMPLETION_RETRIES:
            retry_count += 1
            self.logger.log(
                AgentRole.ORCHESTRATOR,
                f"Response incomplete (attempt {retry_count}/{self.MAX_COMPLETION_RETRIES}) "
                f"— requesting continuation: {evaluation.incompleteness_details}",
            )

            try:
                response = self.execution.continue_response(
                    user_input=user_input,
                    partial_response=response.content,
                    incompleteness_details=evaluation.incompleteness_details,
                )

                # Re-evaluate the continued response
                evaluation = self.evaluation.run(
                    user_input=user_input,
                    perception=perception,
                    strategy=strategy,
                    response=response,
                )
            except Exception as retry_exc:
                self.logger.error(
                    AgentRole.ORCHESTRATOR,
                    f"Continuation attempt {retry_count} failed: {retry_exc}",
                )
                break  # Keep the best response we have so far

        if retry_count > 0:
            self.logger.log(
                AgentRole.ORCHESTRATOR,
                f"Completion retries used: {retry_count} — "
                f"final status: {evaluation.task_completion}, "
                f"complete: {evaluation.is_complete}",
            )

        # ── Phase 5: LEARN ──
        # Extract lessons, store episode, update semantic & procedural memory,
        # and flag domains for knowledge synthesis when threshold is reached
        user_text = user_input.text or "(non-text input)"
        learning = self.learning.run(
            user_input_text=user_text,
            perception=perception,
            strategy=strategy,
            evaluation=evaluation,
        )

        # ── Phase 6: META-REFLECT (periodic — every N interactions) ──
        # Higher-order self-analysis: detect biases, track improvement trends,
        # generate evolution plan.  Only fires at the configured interval.
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

        # ── Periodic consolidation (temporal-axis promotion) ──
        # Every 10 interactions, promote high-value long-term memories
        # to persistent tier so they are never pruned.
        if self._interaction_count % 10 == 0 and self._interaction_count > 0:
            promoted = self.memory.consolidate()
            if any(v > 0 for v in promoted.values()):
                self.logger.log(
                    AgentRole.ORCHESTRATOR,
                    f"Memory consolidation: {promoted}",
                )

        # Store assistant response in working memory
        self.memory.working.add_conversation_turn(
            "assistant", response.content[:500]
        )

        self.logger.separator()
        self.logger.log(
            AgentRole.ORCHESTRATOR,
            f"Interaction #{self._interaction_count} complete — "
            f"quality: {evaluation.output_quality}/10",
        )

        return response

    # ------------------------------------------------------------------
    # Internal: Parallel Modality Preprocessing
    # ------------------------------------------------------------------

    def _preprocess_modalities(self, user_input: MultimodalInput) -> dict[str, str]:
        """Pre-process non-text modalities in parallel.

        When the user sends images + audio + video simultaneously, each
        modality processor runs in its own thread.  The resulting text
        descriptions are returned as a dict keyed by modality name so
        downstream agents can incorporate them.
        """
        tasks: dict[str, tuple[Any, dict[str, Any]]] = {}

        if user_input.image_paths:
            tasks["vision"] = (
                self.vision_proc,
                {"task": "Describe the image(s) in detail", "image_paths": user_input.image_paths},
            )
        if user_input.audio_paths:
            tasks["audio"] = (
                self.audio_proc,
                {"task": "Transcribe and analyse the audio", "audio_paths": user_input.audio_paths},
            )
        if user_input.video_paths:
            tasks["video"] = (
                self.video_proc,
                {
                    "task": "Analyse the video content",
                    "video_paths": user_input.video_paths,
                    "vision_processor": self.vision_proc,
                    "audio_processor": self.audio_proc,
                },
            )

        if not tasks:
            return {}

        # Sequential or parallel depending on config & worker count
        n_tasks = len(tasks)
        use_parallel = self._parallel_modalities and self._max_workers > 1 and n_tasks > 1

        if use_parallel:
            self.logger.log(
                AgentRole.ORCHESTRATOR,
                f"Parallel modality preprocessing: {list(tasks.keys())} "
                f"({self._max_workers} workers)",
            )
            results: dict[str, str] = {}
            with ThreadPoolExecutor(
                max_workers=min(self._max_workers, n_tasks),
                thread_name_prefix="modality",
            ) as pool:
                futures = {
                    pool.submit(proc.run, **kwargs): name
                    for name, (proc, kwargs) in tasks.items()
                }
                for fut in as_completed(futures):
                    name = futures[fut]
                    try:
                        results[name] = fut.result()
                    except Exception as exc:
                        self.logger.error(
                            AgentRole.ORCHESTRATOR,
                            f"Modality '{name}' failed: {exc}",
                        )
                        results[name] = f"(error processing {name})"
            return results

        # Sequential fallback
        self.logger.log(
            AgentRole.ORCHESTRATOR,
            f"Sequential modality preprocessing: {list(tasks.keys())}",
        )
        results = {}
        for name, (proc, kwargs) in tasks.items():
            try:
                results[name] = proc.run(**kwargs)
            except Exception as exc:
                self.logger.error(
                    AgentRole.ORCHESTRATOR,
                    f"Modality '{name}' failed: {exc}",
                )
                results[name] = f"(error processing {name})"
        return results
