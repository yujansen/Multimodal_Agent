"""Procedural Memory — reusable action sequences and decision trees.

Stores refined procedures that can be recalled and executed for recurring
task types, including multi-step reasoning chains, tool-use protocols,
error recovery procedures, and modality-specific processing pipelines.

Performance: maintains an inverted index by task_type so that
``find_by_task_type`` and ``best_procedure`` run in O(k log k) instead
of O(n).
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pinocchio.models.enums import MemoryTier, TaskType
from pinocchio.models.schemas import ProceduralEntry


class ProceduralMemory:
    """Persistent procedural memory store backed by a JSON file.

    Skills / Capabilities:
      - Store and retrieve reusable action templates / procedures
      - Look up the best procedure for a given task type
      - Track procedure success rates and usage counts
      - Refine procedures based on evaluation feedback
      - Rank procedures by effectiveness
      - Persist to disk for cross-session continuity
      - O(1) lookup by entry_id via hash index
      - O(k log k) search by task_type via inverted index
    """

    def __init__(self, storage_path: str = "data/procedural_memory.json") -> None:
        self._path = Path(storage_path)
        self._entries: list[ProceduralEntry] = []
        # ── Indices ──
        self._id_index: dict[str, ProceduralEntry] = {}
        self._task_index: dict[TaskType, list[ProceduralEntry]] = defaultdict(list)
        self._load()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _index_entry(self, entry: ProceduralEntry) -> None:
        self._id_index[entry.entry_id] = entry
        self._task_index[entry.task_type].append(entry)

    def _rebuild_indices(self) -> None:
        self._id_index.clear()
        self._task_index.clear()
        for e in self._entries:
            self._index_entry(e)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    raw: list[dict[str, Any]] = json.load(f)
                self._entries = [ProceduralEntry.from_dict(d) for d in raw]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                import shutil
                backup = self._path.with_suffix(".json.bak")
                shutil.copy2(self._path, backup)
                self._entries = []
        self._rebuild_indices()

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self._entries], f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, entry: ProceduralEntry) -> None:
        self._entries.append(entry)
        self._index_entry(entry)
        self.save()

    def get(self, entry_id: str) -> ProceduralEntry | None:
        return self._id_index.get(entry_id)

    def all(self) -> list[ProceduralEntry]:
        return list(self._entries)

    @property
    def count(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def find_by_task_type(self, task_type: TaskType) -> list[ProceduralEntry]:
        """Return all procedures for a given task type, ranked by success rate (index-accelerated)."""
        matches = list(self._task_index.get(task_type, []))
        matches.sort(key=lambda e: e.success_rate, reverse=True)
        return matches

    def best_procedure(self, task_type: TaskType) -> ProceduralEntry | None:
        """Return the single best procedure for a task type (highest success rate)."""
        procedures = self.find_by_task_type(task_type)
        return procedures[0] if procedures else None

    def search_by_name(self, name: str) -> list[ProceduralEntry]:
        name_lower = name.lower()
        return [e for e in self._entries if name_lower in e.name.lower()]

    # ------------------------------------------------------------------
    # Update & Refinement
    # ------------------------------------------------------------------

    def record_usage(self, entry_id: str, success: bool) -> None:
        """Record a usage of a procedure and update its success rate."""
        entry = self.get(entry_id)
        if entry is None:
            return
        total = entry.usage_count
        old_rate = entry.success_rate
        entry.usage_count = total + 1
        # Incremental average update
        entry.success_rate = (old_rate * total + (1.0 if success else 0.0)) / (total + 1)
        entry.last_used = datetime.now(timezone.utc).isoformat()
        self.save()

    def refine_steps(self, entry_id: str, new_steps: list[str]) -> None:
        """Replace the steps of a procedure with an improved version."""
        entry = self.get(entry_id)
        if entry:
            entry.steps = new_steps
            self.save()

    def top_procedures(self, limit: int = 5) -> list[ProceduralEntry]:
        """Return top procedures globally by success rate (min 2 uses)."""
        qualified = [e for e in self._entries if e.usage_count >= 2]
        qualified.sort(key=lambda e: e.success_rate, reverse=True)
        return qualified[:limit]

    # ------------------------------------------------------------------
    # Tier-aware queries (temporal axis)
    # ------------------------------------------------------------------

    def get_by_tier(self, tier: MemoryTier) -> list[ProceduralEntry]:
        """Return all procedures at a given temporal tier."""
        return [e for e in self._entries if e.memory_tier == tier]

    def get_persistent(self) -> list[ProceduralEntry]:
        """Return proven procedures that are permanently stored."""
        return self.get_by_tier(MemoryTier.PERSISTENT)

    def promote_to_persistent(self, entry_id: str) -> bool:
        """Promote a procedure to persistent tier (never pruned)."""
        entry = self.get(entry_id)
        if entry is None:
            return False
        entry.memory_tier = MemoryTier.PERSISTENT
        self.save()
        return True

    def consolidate_proven(self, min_uses: int = 5, min_rate: float = 0.8) -> list[ProceduralEntry]:
        """Identify long-term procedures that should be promoted to persistent.

        Procedures with high usage count and success rate are candidates
        for permanent retention.
        """
        return [
            e for e in self._entries
            if e.memory_tier == MemoryTier.LONG_TERM
            and e.usage_count >= min_uses
            and e.success_rate >= min_rate
        ]
