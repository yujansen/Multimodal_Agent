"""Semantic Memory — distilled, generalizable knowledge.

Represents abstracted knowledge extracted from multiple episodic experiences:
domain heuristics, cross-modal reasoning patterns, user preference models,
and effective strategy templates.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pinocchio.models.schemas import SemanticEntry


class SemanticMemory:
    """Persistent semantic memory store backed by a JSON file.

    Skills / Capabilities:
      - Store generalizable knowledge entries with domain tags
      - Retrieve knowledge by domain or keyword
      - Update confidence scores as more evidence accumulates
      - Merge duplicate knowledge entries
      - Trigger knowledge synthesis when episode threshold is reached
      - Persist to disk for cross-session continuity
    """

    SYNTHESIS_THRESHOLD = 10  # episodes per domain before synthesis

    def __init__(self, storage_path: str = "data/semantic_memory.json") -> None:
        self._path = Path(storage_path)
        self._entries: list[SemanticEntry] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                raw: list[dict[str, Any]] = json.load(f)
            self._entries = [SemanticEntry.from_dict(d) for d in raw]

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self._entries], f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, entry: SemanticEntry) -> None:
        self._entries.append(entry)
        self.save()

    def get(self, entry_id: str) -> SemanticEntry | None:
        for e in self._entries:
            if e.entry_id == entry_id:
                return e
        return None

    def all(self) -> list[SemanticEntry]:
        return list(self._entries)

    @property
    def count(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search_by_domain(self, domain: str, limit: int = 10) -> list[SemanticEntry]:
        """Return entries matching a domain (case-insensitive substring)."""
        domain_lower = domain.lower()
        return [e for e in self._entries if domain_lower in e.domain.lower()][:limit]

    def search_by_keyword(self, keyword: str, limit: int = 10) -> list[SemanticEntry]:
        keyword_lower = keyword.lower()
        return [
            e
            for e in self._entries
            if keyword_lower in e.knowledge.lower() or keyword_lower in e.domain.lower()
        ][:limit]

    def get_high_confidence(self, threshold: float = 0.7) -> list[SemanticEntry]:
        """Return entries with confidence above a threshold."""
        return [e for e in self._entries if e.confidence >= threshold]

    # ------------------------------------------------------------------
    # Update & Merge
    # ------------------------------------------------------------------

    def update_confidence(self, entry_id: str, new_confidence: float) -> None:
        entry = self.get(entry_id)
        if entry:
            entry.confidence = max(0.0, min(1.0, new_confidence))
            entry.updated_at = datetime.now(timezone.utc).isoformat()
            self.save()

    def add_source_episode(self, entry_id: str, episode_id: str) -> None:
        entry = self.get(entry_id)
        if entry and episode_id not in entry.source_episodes:
            entry.source_episodes.append(episode_id)
            entry.updated_at = datetime.now(timezone.utc).isoformat()
            self.save()

    def domain_entry_count(self, domain: str) -> int:
        return len(self.search_by_domain(domain))

    def needs_synthesis(self, domain: str, episode_count: int) -> bool:
        """Check if domain has accumulated enough episodes for knowledge synthesis."""
        return episode_count >= self.SYNTHESIS_THRESHOLD
