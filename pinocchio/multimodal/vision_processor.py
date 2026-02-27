"""VisionProcessor — sub-agent for image modality processing.

Handles image understanding, description, and cross-modal reasoning
between visual and textual information.

Skills / Capabilities
─────────────────────
1. **Image Understanding**
   Analyse an image and produce a rich, contextual description covering
   objects, scene layout, colours, text (OCR), and visual semantics.

2. **Visual Question Answering (VQA)**
   Answer arbitrary natural-language questions about the content of an
   image.

3. **OCR / Text Extraction**
   Extract any text visible in an image, including handwritten text,
   signs, labels, and screen content.

4. **Image Comparison**
   Compare two or more images and describe their differences and
   similarities.

5. **Image → Text Captioning**
   Generate concise or detailed captions suitable for accessibility,
   documentation, or downstream reasoning.

6. **Visual Reasoning**
   Perform multi-step reasoning over visual content — e.g., counting
   objects, inferring relationships, spatial reasoning.

7. **Chart / Diagram Interpretation**
   Understand charts, graphs, diagrams, and infographics, extracting
   both data points and high-level takeaways.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from pinocchio.agents.base_agent import BaseAgent
from pinocchio.models.enums import AgentRole

_SYSTEM_PROMPT = """\
You are the Vision Processor of Pinocchio, a self-evolving multimodal AI.
You specialise in image understanding and visual reasoning.
Analyse the provided image(s) carefully and complete the requested task.
"""


class VisionProcessor(BaseAgent):
    """Processes image-modality tasks via the LLM vision API."""

    role = AgentRole.VISION_PROCESSOR

    @staticmethod
    def _encode_image(path: str) -> str:
        """Encode a local image file as a base64 data URL."""
        p = Path(path)
        suffix = p.suffix.lower().lstrip(".")
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(suffix, "png")
        data = base64.b64encode(p.read_bytes()).decode()
        return f"data:image/{mime};base64,{data}"

    def run(self, *, task: str, image_paths: list[str], **kwargs: Any) -> str:  # type: ignore[override]
        """Execute a vision task on one or more images.

        Parameters
        ----------
        task : Description of the vision task (e.g., "describe this image",
               "what text is in this image?").
        image_paths : File paths or URLs of images to process.
        """
        self._log(f"Vision task: {task} — {len(image_paths)} image(s)")

        urls: list[str] = []
        for p in image_paths:
            if p.startswith(("http://", "https://", "data:")):
                urls.append(p)
            else:
                urls.append(self._encode_image(p))

        vision_msg = self.llm.build_vision_message(
            f"Task: {task}", urls
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            vision_msg,
        ]
        result = self.llm.chat(messages)
        self._log(f"Vision processing complete — {len(result)} chars")
        return result
