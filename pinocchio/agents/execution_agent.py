"""ExecutionAgent — Phase 3 of the cognitive loop (EXECUTE).

Carries out the task according to the strategy produced in Phase 2.
This is the agent that generates the actual user-facing response.  It follows
the strategic plan step-by-step, monitors intermediate results, and triggers
adaptive re-planning when something goes wrong.

Skills / Capabilities
─────────────────────
1. **Plan Execution**
   Follow the strategy plan step by step, generating intermediate outputs
   and assembling them into a coherent final result.

2. **Adaptive Re-Planning**
   If an intermediate step produces unexpected or low-quality output, pause
   and switch to the fallback plan rather than blindly continuing.

3. **Multimodal Output Assembly**
   Construct responses that may combine text, images, and other modalities,
   ensuring cross-modal consistency.

4. **Quality Monitoring**
   Continuously assess intermediate results against expected standards
   and adjust approach in real-time.

5. **Tool Invocation**
   When the strategy calls for external tool use (web search, code execution,
   image generation, etc.), dispatch the appropriate tool call and incorporate
   the result.

6. **Context Window Management**
   Efficiently manage the LLM context by summarising prior steps and retaining
   only the most relevant information.

7. **Cross-Modal Consistency Enforcement**
   When producing multimodal output, verify that information across modalities
   is coherent and non-contradictory.
"""

from __future__ import annotations

from typing import Any

from pinocchio.agents.base_agent import BaseAgent
from pinocchio.models.enums import AgentRole
from pinocchio.models.schemas import (
    MultimodalInput,
    PerceptionResult,
    StrategyResult,
    AgentMessage,
)

_SYSTEM_PROMPT = """\
You are the Execution sub-agent of Pinocchio, a self-evolving multimodal AI.
Your task is to produce the BEST possible response to the user's request.

You are given:
1. The original user input
2. A perception analysis
3. A strategy plan

Follow the strategy's modality pipeline.  If an intermediate step seems to
fail, note the issue and switch to the fallback plan.

Your output must be the final, polished response to the user.
Write in the same language the user uses.
Be thorough, accurate, and helpful.
"""


class ExecutionAgent(BaseAgent):
    """Generates the user-facing response following the strategy plan."""

    role = AgentRole.EXECUTION

    def run(  # type: ignore[override]
        self,
        user_input: MultimodalInput,
        perception: PerceptionResult,
        strategy: StrategyResult,
        **kwargs: Any,
    ) -> AgentMessage:
        self.logger.phase("Phase 3: EXECUTE 执行")
        self._log(f"Executing strategy: {strategy.selected_strategy}")

        user_text = user_input.text or "(non-text input)"

        context_prompt = (
            f"=== PERCEPTION ===\n"
            f"Task type: {perception.task_type.value}\n"
            f"Complexity: {perception.complexity.value}/5\n"
            f"Modalities: {[m.value for m in perception.modalities]}\n"
            f"Ambiguities: {perception.ambiguities}\n\n"
            f"=== STRATEGY ===\n"
            f"Strategy: {strategy.selected_strategy}\n"
            f"Pipeline: {strategy.modality_pipeline}\n"
            f"Fusion: {strategy.fusion_strategy.value}\n"
            f"Risk: {strategy.risk_assessment}\n"
            f"Fallback: {strategy.fallback_plan}\n\n"
            f"=== USER INPUT ===\n{user_text}"
        )

        # Vision path: if images are present, build a multimodal message
        if user_input.image_paths:
            self._log("Building multimodal (vision) request…")
            vision_msg = self.llm.build_vision_message(context_prompt, user_input.image_paths)
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                vision_msg,
            ]
            response_text = self.llm.chat(messages)
        else:
            response_text = self.llm.ask(system=_SYSTEM_PROMPT, user=context_prompt)

        self._log(f"Execution complete — response length: {len(response_text)} chars")

        return AgentMessage(
            role="assistant",
            content=response_text,
            confidence=0.8 if strategy.is_novel else 0.9,
            metadata={
                "strategy": strategy.selected_strategy,
                "is_novel_strategy": strategy.is_novel,
            },
        )
