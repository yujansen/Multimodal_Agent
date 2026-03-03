"""Memory Manager — unified façade over the dual-axis memory system.

Pinocchio's memory has two orthogonal classification axes:

**Content axis** (what is stored):
  - Episodic  — concrete past interaction traces
  - Semantic  — distilled, generalisable knowledge
  - Procedural — reusable action templates & strategies

**Temporal axis** (how long it lives):
  - Working    — current session only; volatile, capacity-limited
  - Long-term  — cross-session; persisted to disk, subject to decay
  - Persistent — permanent; core knowledge, never pruned

This manager coordinates reads/writes across all stores and provides
consolidation logic that promotes entries between temporal tiers.
"""

from __future__ import annotations

from pinocchio.memory.episodic_memory import EpisodicMemory
from pinocchio.memory.semantic_memory import SemanticMemory
from pinocchio.memory.procedural_memory import ProceduralMemory
from pinocchio.memory.working_memory import WorkingMemory
from pinocchio.models.enums import MemoryTier, Modality, TaskType
from pinocchio.models.schemas import (
    EpisodicRecord,
    SemanticEntry,
    ProceduralEntry,
)


class MemoryManager:
    """Unified façade for Pinocchio's dual-axis memory system.

    Content axis:  episodic · semantic · procedural
    Temporal axis: working  · long-term · persistent

    Skills / Capabilities:
      - Coordinate reads/writes across all memory stores
      - Provide a single *recall* method that searches all stores
      - Manage working memory for current-session context
      - Trigger knowledge synthesis when a domain reaches the episode threshold
      - Consolidate memories: working→long-term, long-term→persistent
      - Compute cross-memory analytics (improvement trends, strategy rankings)
      - Ensure data consistency across memory stores
    """

    def __init__(self, data_dir: str = "data") -> None:
        # ── Content-axis stores (persistent on disk) ──
        self.episodic = EpisodicMemory(f"{data_dir}/episodic_memory.json")
        self.semantic = SemanticMemory(f"{data_dir}/semantic_memory.json")
        self.procedural = ProceduralMemory(f"{data_dir}/procedural_memory.json")

        # ── Temporal-axis: working memory (volatile, in-RAM) ──
        self.working = WorkingMemory(capacity=50)

        # Domains pending synthesis — consumed by LearningAgent
        self._pending_synthesis: list[str] = []

    # ------------------------------------------------------------------
    # Unified recall (both axes)
    # ------------------------------------------------------------------

    def recall(
        self,
        task_type: TaskType,
        modalities: list[Modality],
        keyword: str = "",
    ) -> dict:
        """Retrieve relevant context from all memory stores at once.

        Returns a dict with:
          - similar_episodes: list[EpisodicRecord]
          - relevant_knowledge: list[SemanticEntry]
          - best_procedure: ProceduralEntry | None
          - recent_lessons: list[str]
          - persistent_knowledge: list[SemanticEntry]
          - working_context: str
        """
        similar = self.episodic.find_similar(task_type, modalities, limit=3)
        knowledge: list[SemanticEntry] = []
        if keyword:
            knowledge = self.semantic.search_by_keyword(keyword, limit=5)
        else:
            knowledge = self.semantic.search_by_domain(task_type.value, limit=5)
        procedure = self.procedural.best_procedure(task_type)
        lessons = self.episodic.recent_lessons(limit=5)

        # Temporal-axis enrichment
        persistent_knowledge = self.semantic.get_persistent()[:5]
        working_context = self.working.format_conversation_context(max_turns=5)

        return {
            "similar_episodes": similar,
            "relevant_knowledge": knowledge,
            "best_procedure": procedure,
            "recent_lessons": lessons,
            "persistent_knowledge": persistent_knowledge,
            "working_context": working_context,
        }

    # ------------------------------------------------------------------
    # Cross-memory operations
    # ------------------------------------------------------------------

    def store_episode(self, episode: EpisodicRecord) -> None:
        """Store an episode and flag the domain for synthesis if threshold is reached."""
        self.episodic.add(episode)
        domain = episode.task_type.value
        domain_episodes = self.episodic.search_by_task_type(episode.task_type, limit=100)
        if self.semantic.needs_synthesis(domain, len(domain_episodes)):
            if domain not in self._pending_synthesis:
                self._pending_synthesis.append(domain)

    def pop_pending_synthesis(self) -> list[str]:
        """Return and clear domains that are ready for knowledge synthesis.

        This is consumed by the LearningAgent after each interaction to
        decide whether to trigger cross-episode synthesis for specific domains.
        """
        domains = self._pending_synthesis.copy()
        self._pending_synthesis.clear()
        return domains

    def store_knowledge(self, entry: SemanticEntry) -> None:
        self.semantic.add(entry)

    def store_procedure(self, entry: ProceduralEntry) -> None:
        self.procedural.add(entry)

    def record_procedure_usage(self, entry_id: str, success: bool) -> None:
        self.procedural.record_usage(entry_id, success)

    # ------------------------------------------------------------------
    # Temporal-axis: consolidation & promotion
    # ------------------------------------------------------------------

    def consolidate(self) -> dict[str, int]:
        """Run consolidation across all content stores.

        Promotes high-value long-term entries to persistent tier.
        Returns counts of promoted entries per store.
        """
        promoted: dict[str, int] = {"episodic": 0, "semantic": 0, "procedural": 0}

        # Episodic: promote high-score episodes with lessons
        for ep in self.episodic.consolidate_high_value():
            ep.memory_tier = MemoryTier.PERSISTENT
            promoted["episodic"] += 1
        if promoted["episodic"]:
            self.episodic.save()

        # Semantic: promote high-confidence, well-sourced knowledge
        for entry in self.semantic.consolidate_high_confidence():
            entry.memory_tier = MemoryTier.PERSISTENT
            promoted["semantic"] += 1
        if promoted["semantic"]:
            self.semantic.save()

        # Procedural: promote proven procedures
        for entry in self.procedural.consolidate_proven():
            entry.memory_tier = MemoryTier.PERSISTENT
            promoted["procedural"] += 1
        if promoted["procedural"]:
            self.procedural.save()

        return promoted

    def reset_working_memory(self) -> None:
        """Clear the volatile working memory (session reset)."""
        self.working.clear()

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def improvement_trend(self, window: int = 10) -> dict:
        """Return improvement metrics over a sliding window."""
        all_eps = self.episodic.all()
        if len(all_eps) < window:
            return {"recent_avg": self.episodic.average_score(), "trend": "insufficient_data"}
        recent = self.episodic.average_score(last_n=window)
        older = self.episodic.average_score(last_n=len(all_eps)) if len(all_eps) > window else recent
        if recent > older + 0.5:
            trend = "improving"
        elif recent < older - 0.5:
            trend = "declining"
        else:
            trend = "stable"
        return {"recent_avg": round(recent, 2), "older_avg": round(older, 2), "trend": trend}

    def summary(self) -> dict:
        """High-level summary of all memory stores (both axes)."""
        return {
            # Content axis
            "episodic_count": self.episodic.count,
            "semantic_count": self.semantic.count,
            "procedural_count": self.procedural.count,
            "avg_score": round(self.episodic.average_score(), 2),
            "error_frequency": self.episodic.error_frequency(),
            "top_procedures": [
                {"name": p.name, "success_rate": round(p.success_rate, 2)}
                for p in self.procedural.top_procedures(limit=3)
            ],
            # Temporal axis
            "working_memory": self.working.summary(),
            "persistent_episodes": len(self.episodic.get_persistent()),
            "persistent_knowledge": len(self.semantic.get_persistent()),
            "persistent_procedures": len(self.procedural.get_persistent()),
        }
