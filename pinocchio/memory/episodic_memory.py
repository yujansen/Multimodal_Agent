"""Episodic Memory — structured log of past interactions.

Each episode captures the full trace of a single interaction: what was asked,
how the agent reasoned, what strategy it used, outcome quality, and lessons
learned.  This memory allows Pinocchio to recall specific past experiences
and apply relevant lessons to new tasks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pinocchio.models.enums import Modality, TaskType
from pinocchio.models.schemas import EpisodicRecord


class EpisodicMemory:
    """Persistent episodic memory store backed by a JSON file.

    Skills / Capabilities:
      - Store and retrieve full interaction episodes
      - Search episodes by task type, modality, or keyword
      - Return the *k* most similar episodes to a given query
      - Compute aggregate statistics (avg score, error frequency)
      - Persist to disk for cross-session continuity
    """

    def __init__(self, storage_path: str = "data/episodic_memory.json") -> None:
        self._path = Path(storage_path)
        self._episodes: list[EpisodicRecord] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                raw: list[dict[str, Any]] = json.load(f)
            self._episodes = [EpisodicRecord.from_dict(d) for d in raw]

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self._episodes], f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, episode: EpisodicRecord) -> None:
        """Add a new episode to memory and persist."""
        self._episodes.append(episode)
        self.save()

    def get(self, episode_id: str) -> EpisodicRecord | None:
        for ep in self._episodes:
            if ep.episode_id == episode_id:
                return ep
        return None

    def all(self) -> list[EpisodicRecord]:
        return list(self._episodes)

    @property
    def count(self) -> int:
        return len(self._episodes)

    # ------------------------------------------------------------------
    # Query / Search
    # ------------------------------------------------------------------

    def search_by_task_type(self, task_type: TaskType, limit: int = 5) -> list[EpisodicRecord]:
        """Return most recent episodes of a given task type."""
        matches = [e for e in self._episodes if e.task_type == task_type]
        return matches[-limit:]

    def search_by_modality(self, modality: Modality, limit: int = 5) -> list[EpisodicRecord]:
        """Return most recent episodes involving a given modality."""
        matches = [e for e in self._episodes if modality in e.modalities]
        return matches[-limit:]

    def search_by_keyword(self, keyword: str, limit: int = 5) -> list[EpisodicRecord]:
        """Simple keyword search across intent, strategy, lessons, and notes."""
        keyword_lower = keyword.lower()
        matches: list[EpisodicRecord] = []
        for ep in self._episodes:
            text_blob = " ".join(
                [ep.user_intent, ep.strategy_used, ep.improvement_notes]
                + ep.lessons
                + ep.error_patterns
            ).lower()
            if keyword_lower in text_blob:
                matches.append(ep)
        return matches[-limit:]

    def find_similar(
        self,
        task_type: TaskType,
        modalities: list[Modality],
        limit: int = 3,
    ) -> list[EpisodicRecord]:
        """Find the most similar past episodes by task type + modality overlap.

        Similarity heuristic:
          +2 for matching task type
          +1 for each shared modality
        Returns top-k by score, then by recency.
        """

        def _score(ep: EpisodicRecord) -> int:
            s = 0
            if ep.task_type == task_type:
                s += 2
            s += len(set(ep.modalities) & set(modalities))
            return s

        scored = [(ep, _score(ep)) for ep in self._episodes if _score(ep) > 0]
        scored.sort(key=lambda x: (x[1], x[0].timestamp), reverse=True)
        return [ep for ep, _ in scored[:limit]]

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def average_score(self, last_n: int | None = None) -> float:
        """Average outcome score over the last *n* episodes (or all)."""
        episodes = self._episodes[-last_n:] if last_n else self._episodes
        if not episodes:
            return 0.0
        return sum(e.outcome_score for e in episodes) / len(episodes)

    def error_frequency(self) -> dict[str, int]:
        """Count of each error pattern across all episodes."""
        freq: dict[str, int] = {}
        for ep in self._episodes:
            for err in ep.error_patterns:
                freq[err] = freq.get(err, 0) + 1
        return freq

    def recent_lessons(self, limit: int = 10) -> list[str]:
        """Collect the most recent lessons across episodes."""
        lessons: list[str] = []
        for ep in reversed(self._episodes):
            lessons.extend(ep.lessons)
            if len(lessons) >= limit:
                break
        return lessons[:limit]
