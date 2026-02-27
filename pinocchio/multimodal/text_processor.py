"""TextProcessor — sub-agent for text modality processing.

Handles all text-centric tasks including understanding, generation, and
cross-modal text translation.

Skills / Capabilities
─────────────────────
1. **Text Understanding**
   Deep comprehension of user queries — intent extraction, sentiment analysis,
   named entity recognition, and semantic parsing.

2. **Text Generation**
   Produce high-quality output text in any style and format: answers,
   summaries, analyses, creative writing, code, etc.

3. **Language Detection & Translation**
   Detect the language of input text and translate between languages
   while preserving nuance and tone.

4. **Summarisation**
   Condense long documents into concise summaries at varying levels of
   detail (abstractive and extractive).

5. **Text → Other Modality Bridging**
   Generate structured descriptions / captions suitable for feeding into
   image, audio, or video generation pipelines.

6. **Semantic Similarity Scoring**
   Compare two text segments and estimate their semantic overlap, useful
   for deduplication and relevance ranking.

7. **Structured Information Extraction**
   Parse unstructured text into structured formats (JSON, tables, key-value
   pairs) based on user-specified schemas.
"""

from __future__ import annotations

from typing import Any

from pinocchio.agents.base_agent import BaseAgent
from pinocchio.models.enums import AgentRole

_SYSTEM_PROMPT = """\
You are the Text Processor of Pinocchio, a self-evolving multimodal AI.
You specialise in text understanding and generation.
Process the following text task and return your result.
Write in the same language as the input unless instructed otherwise.
"""


class TextProcessor(BaseAgent):
    """Processes text-modality tasks."""

    role = AgentRole.TEXT_PROCESSOR

    def run(self, *, task: str, text: str, **kwargs: Any) -> str:  # type: ignore[override]
        """Execute a text processing task.

        Parameters
        ----------
        task : Description of the requested text operation (e.g., "summarise",
               "translate to English", "extract entities").
        text : The input text to process.
        """
        self._log(f"Text task: {task}")
        prompt = f"Task: {task}\n\nInput text:\n{text}"
        result = self.llm.ask(system=_SYSTEM_PROMPT, user=prompt)
        self._log(f"Text processing complete — {len(result)} chars")
        return result
