"""EvaluationAgent — Phase 4 of the cognitive loop (EVALUATE).

Performs rigorous self-assessment after the Execution phase completes.
It scores the output quality, strategy effectiveness, cross-modal coherence,
and identifies what went well, what went wrong, and any surprise factors.

Skills / Capabilities
─────────────────────
1. **Output Quality Scoring**
   Rate the response quality on a 1–10 scale considering accuracy,
   completeness, clarity, and helpfulness.

2. **Strategy Effectiveness Scoring**
   Evaluate how well the chosen strategy served the task: was it efficient?
   Did it lead to a good result on the first attempt?

3. **Cross-Modal Coherence Assessment**
   For multimodal outputs, check if information across modalities is consistent
   and mutually reinforcing.

4. **Success / Failure Analysis**
   Enumerate specific things that went well and things that went wrong
   during execution.

5. **Surprise Factor Identification**
   Flag unexpected elements that were encountered during processing — these
   are high-value learning signals.

6. **User Satisfaction Inference**
   Based on the user's subsequent feedback (if available), infer satisfaction
   level; otherwise mark as "awaiting".

7. **Completion Status Classification**
   Determine whether the task was fully completed, partially addressed, or
   failed entirely.
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
    EvaluationResult,
)

_SYSTEM_PROMPT = """\
You are the Evaluation sub-agent of Pinocchio, a self-evolving multimodal AI.
Your job is to rigorously evaluate the quality of a response that was just
generated.

You are given:
1. The user's original input
2. Perception analysis
3. Strategy used
4. The actual response produced

Output valid JSON with exactly these keys:
{
  "task_completion": "complete" or "partial" or "failed",
  "output_quality": integer 1-10,
  "strategy_effectiveness": integer 1-10,
  "went_well": ["list of positives"],
  "went_wrong": ["list of negatives or empty"],
  "surprises": ["unexpected elements encountered or empty"],
  "cross_modal_coherence": integer 1-10 (use 5 if single modality),
  "analysis": "free-text evaluation summary (2-4 sentences)"
}

Be honest and critical.  Over-generous ratings reduce learning quality.
"""


class EvaluationAgent(BaseAgent):
    """Evaluates the execution output and produces an EvaluationResult."""

    role = AgentRole.EVALUATION

    def run(  # type: ignore[override]
        self,
        user_input: MultimodalInput,
        perception: PerceptionResult,
        strategy: StrategyResult,
        response: AgentMessage,
        **kwargs: Any,
    ) -> EvaluationResult:
        self.logger.phase("Phase 4: EVALUATE 评估")
        self._log("Evaluating output quality…")

        user_text = user_input.text or "(non-text input)"
        eval_prompt = (
            f"=== USER INPUT ===\n{user_text}\n\n"
            f"=== PERCEPTION ===\n"
            f"Task type: {perception.task_type.value}\n"
            f"Complexity: {perception.complexity.value}/5\n\n"
            f"=== STRATEGY ===\n"
            f"Strategy: {strategy.selected_strategy}\n"
            f"Pipeline: {strategy.modality_pipeline}\n\n"
            f"=== RESPONSE ===\n{response.content[:3000]}"
        )

        llm_result = self.llm.ask_json(system=_SYSTEM_PROMPT, user=eval_prompt)

        result = EvaluationResult(
            task_completion=llm_result.get("task_completion", "complete"),
            output_quality=int(llm_result.get("output_quality", 5)),
            strategy_effectiveness=int(llm_result.get("strategy_effectiveness", 5)),
            went_well=llm_result.get("went_well", []),
            went_wrong=llm_result.get("went_wrong", []),
            surprises=llm_result.get("surprises", []),
            cross_modal_coherence=int(llm_result.get("cross_modal_coherence", 5)),
            user_satisfaction="awaiting",
            raw_analysis=llm_result.get("analysis", ""),
        )

        self._log(
            f"Quality: {result.output_quality}/10 | "
            f"Strategy: {result.strategy_effectiveness}/10 | "
            f"Status: {result.task_completion}"
        )
        if result.went_wrong:
            self._warn(f"Issues found: {result.went_wrong}")

        return result
