"""Comprehensive edge-case tests for the three memory subsystems.

Covers: persistence round-trips, corrupted JSON, limit parameters,
nonexistent IDs, keyword search, analytics, cross-memory operations,
improvement_trend branches, error frequency, knowledge synthesis checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pinocchio.memory.episodic_memory import EpisodicMemory
from pinocchio.memory.semantic_memory import SemanticMemory
from pinocchio.memory.procedural_memory import ProceduralMemory
from pinocchio.memory.memory_manager import MemoryManager
from pinocchio.models.enums import Modality, TaskType
from pinocchio.models.schemas import (
    EpisodicRecord,
    SemanticEntry,
    ProceduralEntry,
)


# =====================================================================
# Episodic Memory
# =====================================================================
class TestEpisodicMemoryEdge:
    """Edge cases for EpisodicMemory."""

    def test_empty_memory_average_score(self, tmp_path):
        em = EpisodicMemory(str(tmp_path / "ep.json"))
        assert em.average_score() == 0.0
        assert em.average_score(last_n=5) == 0.0

    def test_find_similar_empty(self, tmp_path):
        em = EpisodicMemory(str(tmp_path / "ep.json"))
        result = em.find_similar(TaskType.QUESTION_ANSWERING, [Modality.TEXT])
        assert result == []

    def test_search_by_keyword_empty(self, tmp_path):
        em = EpisodicMemory(str(tmp_path / "ep.json"))
        assert em.search_by_keyword("anything") == []

    def test_search_by_task_type_limit(self, tmp_path):
        em = EpisodicMemory(str(tmp_path / "ep.json"))
        for i in range(10):
            em.add(EpisodicRecord(
                task_type=TaskType.QUESTION_ANSWERING,
                user_intent=f"q{i}",
            ))
        result = em.search_by_task_type(TaskType.QUESTION_ANSWERING, limit=3)
        assert len(result) == 3
        # Should be the 3 most recent
        assert result[-1].user_intent == "q9"

    def test_search_by_modality(self, tmp_path):
        em = EpisodicMemory(str(tmp_path / "ep.json"))
        em.add(EpisodicRecord(modalities=[Modality.TEXT], user_intent="text only"))
        em.add(EpisodicRecord(modalities=[Modality.IMAGE], user_intent="image only"))
        em.add(EpisodicRecord(modalities=[Modality.TEXT, Modality.IMAGE], user_intent="both"))
        text_eps = em.search_by_modality(Modality.TEXT)
        assert len(text_eps) == 2
        image_eps = em.search_by_modality(Modality.IMAGE)
        assert len(image_eps) == 2

    def test_search_by_keyword_in_lessons(self, tmp_path):
        em = EpisodicMemory(str(tmp_path / "ep.json"))
        em.add(EpisodicRecord(lessons=["use chain-of-thought"], user_intent="math"))
        em.add(EpisodicRecord(lessons=["be concise"], user_intent="summary"))
        matched = em.search_by_keyword("chain-of-thought")
        assert len(matched) == 1
        assert matched[0].user_intent == "math"

    def test_search_by_keyword_in_error_patterns(self, tmp_path):
        em = EpisodicMemory(str(tmp_path / "ep.json"))
        em.add(EpisodicRecord(error_patterns=["timeout_error"], user_intent="slow"))
        matched = em.search_by_keyword("timeout")
        assert len(matched) == 1

    def test_get_nonexistent_episode(self, tmp_path):
        em = EpisodicMemory(str(tmp_path / "ep.json"))
        assert em.get("nonexistent") is None

    def test_error_frequency_counts_correctly(self, tmp_path):
        em = EpisodicMemory(str(tmp_path / "ep.json"))
        em.add(EpisodicRecord(error_patterns=["timeout", "hallucination"]))
        em.add(EpisodicRecord(error_patterns=["timeout"]))
        em.add(EpisodicRecord(error_patterns=[]))
        freq = em.error_frequency()
        assert freq["timeout"] == 2
        assert freq["hallucination"] == 1

    def test_recent_lessons_collects_from_newest_first(self, tmp_path):
        em = EpisodicMemory(str(tmp_path / "ep.json"))
        em.add(EpisodicRecord(lessons=["old lesson 1"]))
        em.add(EpisodicRecord(lessons=["mid lesson"]))
        em.add(EpisodicRecord(lessons=["new lesson 1", "new lesson 2"]))
        lessons = em.recent_lessons(limit=3)
        assert lessons[0] == "new lesson 1"
        assert len(lessons) == 3

    def test_persistence_roundtrip(self, tmp_path):
        path = str(tmp_path / "ep.json")
        em1 = EpisodicMemory(path)
        ep = EpisodicRecord(
            task_type=TaskType.CODE_GENERATION,
            modalities=[Modality.TEXT],
            user_intent="Write a function",
            strategy_used="step_by_step",
            outcome_score=8,
            lessons=["test first"],
            error_patterns=["off_by_one"],
            improvement_notes="Add edge cases",
        )
        em1.add(ep)
        # Reload from disk
        em2 = EpisodicMemory(path)
        assert em2.count == 1
        loaded = em2.get(ep.episode_id)
        assert loaded is not None
        assert loaded.task_type == TaskType.CODE_GENERATION
        assert loaded.outcome_score == 8
        assert loaded.lessons == ["test first"]

    def test_corrupted_json_raises(self, tmp_path):
        path = tmp_path / "ep.json"
        path.write_text("not valid json!!!")
        with pytest.raises((json.JSONDecodeError, Exception)):
            EpisodicMemory(str(path))

    def test_find_similar_scoring(self, tmp_path):
        """Verify the similarity heuristic: +2 for task, +1 per modality."""
        em = EpisodicMemory(str(tmp_path / "ep.json"))
        # Perfect match: same task + same modalities
        em.add(EpisodicRecord(
            task_type=TaskType.QUESTION_ANSWERING,
            modalities=[Modality.TEXT, Modality.IMAGE],
            user_intent="perfect match",
        ))
        # Partial: same task only
        em.add(EpisodicRecord(
            task_type=TaskType.QUESTION_ANSWERING,
            modalities=[Modality.AUDIO],
            user_intent="task only",
        ))
        # Mismatched
        em.add(EpisodicRecord(
            task_type=TaskType.CREATIVE_WRITING,
            modalities=[Modality.TEXT],
            user_intent="different",
        ))
        similar = em.find_similar(
            TaskType.QUESTION_ANSWERING, [Modality.TEXT, Modality.IMAGE], limit=3
        )
        # Should have 3 results (different has score > 0 due to TEXT overlap)
        intents = [e.user_intent for e in similar]
        assert intents[0] == "perfect match"  # score = 2+2 = 4

    def test_average_score_with_last_n(self, tmp_path):
        em = EpisodicMemory(str(tmp_path / "ep.json"))
        for s in [3, 4, 5, 8, 9, 10]:
            em.add(EpisodicRecord(outcome_score=s))
        # last 3 = [8, 9, 10], avg = 9.0
        assert em.average_score(last_n=3) == 9.0
        # all = [3,4,5,8,9,10], avg = 6.5
        assert abs(em.average_score() - 6.5) < 0.01


# =====================================================================
# Semantic Memory
# =====================================================================
class TestSemanticMemoryEdge:
    """Edge cases for SemanticMemory."""

    def test_empty_search(self, tmp_path):
        sm = SemanticMemory(str(tmp_path / "sem.json"))
        assert sm.search_by_domain("anything") == []
        assert sm.search_by_keyword("anything") == []

    def test_get_nonexistent(self, tmp_path):
        sm = SemanticMemory(str(tmp_path / "sem.json"))
        assert sm.get("nonexistent") is None

    def test_update_confidence_clamps(self, tmp_path):
        sm = SemanticMemory(str(tmp_path / "sem.json"))
        e = SemanticEntry(domain="test", knowledge="k", confidence=0.5)
        sm.add(e)
        sm.update_confidence(e.entry_id, 1.5)
        assert sm.get(e.entry_id).confidence == 1.0
        sm.update_confidence(e.entry_id, -0.3)
        assert sm.get(e.entry_id).confidence == 0.0

    def test_update_confidence_nonexistent(self, tmp_path):
        sm = SemanticMemory(str(tmp_path / "sem.json"))
        sm.update_confidence("nonexistent", 0.9)  # should not raise

    def test_add_source_episode(self, tmp_path):
        sm = SemanticMemory(str(tmp_path / "sem.json"))
        e = SemanticEntry(domain="test", knowledge="k", source_episodes=["ep1"])
        sm.add(e)
        sm.add_source_episode(e.entry_id, "ep2")
        updated = sm.get(e.entry_id)
        assert "ep2" in updated.source_episodes
        assert updated.updated_at != ""

    def test_add_source_episode_dedup(self, tmp_path):
        sm = SemanticMemory(str(tmp_path / "sem.json"))
        e = SemanticEntry(domain="test", knowledge="k", source_episodes=["ep1"])
        sm.add(e)
        sm.add_source_episode(e.entry_id, "ep1")  # duplicate
        assert len(sm.get(e.entry_id).source_episodes) == 1

    def test_add_source_episode_nonexistent(self, tmp_path):
        sm = SemanticMemory(str(tmp_path / "sem.json"))
        sm.add_source_episode("nonexistent", "ep1")  # should not raise

    def test_domain_entry_count(self, tmp_path):
        sm = SemanticMemory(str(tmp_path / "sem.json"))
        sm.add(SemanticEntry(domain="math", knowledge="k1"))
        sm.add(SemanticEntry(domain="math_advanced", knowledge="k2"))
        sm.add(SemanticEntry(domain="physics", knowledge="k3"))
        assert sm.domain_entry_count("math") == 2  # substring match
        assert sm.domain_entry_count("physics") == 1
        assert sm.domain_entry_count("chemistry") == 0

    def test_get_high_confidence(self, tmp_path):
        sm = SemanticMemory(str(tmp_path / "sem.json"))
        sm.add(SemanticEntry(domain="a", knowledge="k1", confidence=0.9))
        sm.add(SemanticEntry(domain="b", knowledge="k2", confidence=0.3))
        sm.add(SemanticEntry(domain="c", knowledge="k3", confidence=0.7))
        high = sm.get_high_confidence(threshold=0.7)
        assert len(high) == 2

    def test_needs_synthesis(self, tmp_path):
        sm = SemanticMemory(str(tmp_path / "sem.json"))
        assert sm.needs_synthesis("math", 5) is False
        assert sm.needs_synthesis("math", 10) is True
        assert sm.needs_synthesis("math", 15) is True

    def test_persistence_roundtrip(self, tmp_path):
        path = str(tmp_path / "sem.json")
        sm1 = SemanticMemory(path)
        e = SemanticEntry(
            domain="physics",
            knowledge="E=mc^2",
            confidence=0.95,
            source_episodes=["ep1", "ep2"],
        )
        sm1.add(e)
        sm2 = SemanticMemory(path)
        assert sm2.count == 1
        loaded = sm2.get(e.entry_id)
        assert loaded.knowledge == "E=mc^2"
        assert loaded.confidence == 0.95


# =====================================================================
# Procedural Memory
# =====================================================================
class TestProceduralMemoryEdge:
    """Edge cases for ProceduralMemory."""

    def test_empty_best_procedure(self, tmp_path):
        pm = ProceduralMemory(str(tmp_path / "proc.json"))
        assert pm.best_procedure(TaskType.QUESTION_ANSWERING) is None

    def test_get_nonexistent(self, tmp_path):
        pm = ProceduralMemory(str(tmp_path / "proc.json"))
        assert pm.get("nonexistent") is None

    def test_find_by_task_type_sorted(self, tmp_path):
        pm = ProceduralMemory(str(tmp_path / "proc.json"))
        pm.add(ProceduralEntry(
            task_type=TaskType.CODE_GENERATION,
            name="slow",
            success_rate=0.5,
        ))
        pm.add(ProceduralEntry(
            task_type=TaskType.CODE_GENERATION,
            name="fast",
            success_rate=0.9,
        ))
        procs = pm.find_by_task_type(TaskType.CODE_GENERATION)
        assert procs[0].name == "fast"  # highest success_rate first

    def test_find_by_task_type_no_match(self, tmp_path):
        pm = ProceduralMemory(str(tmp_path / "proc.json"))
        pm.add(ProceduralEntry(task_type=TaskType.CODE_GENERATION, name="x"))
        assert pm.find_by_task_type(TaskType.CREATIVE_WRITING) == []

    def test_record_usage_success(self, tmp_path):
        pm = ProceduralMemory(str(tmp_path / "proc.json"))
        p = ProceduralEntry(
            task_type=TaskType.QUESTION_ANSWERING,
            name="qa_proc",
            success_rate=0.8,
            usage_count=4,
        )
        pm.add(p)
        pm.record_usage(p.entry_id, True)
        updated = pm.get(p.entry_id)
        assert updated.usage_count == 5
        # New rate = (0.8*4 + 1) / 5 = 4.2 / 5 = 0.84
        assert abs(updated.success_rate - 0.84) < 0.01
        assert updated.last_used != ""

    def test_record_usage_failure(self, tmp_path):
        pm = ProceduralMemory(str(tmp_path / "proc.json"))
        p = ProceduralEntry(
            task_type=TaskType.QUESTION_ANSWERING,
            name="qa_proc",
            success_rate=0.8,
            usage_count=4,
        )
        pm.add(p)
        pm.record_usage(p.entry_id, False)
        updated = pm.get(p.entry_id)
        assert updated.usage_count == 5
        # New rate = (0.8*4 + 0) / 5 = 3.2 / 5 = 0.64
        assert abs(updated.success_rate - 0.64) < 0.01

    def test_record_usage_nonexistent(self, tmp_path):
        pm = ProceduralMemory(str(tmp_path / "proc.json"))
        pm.record_usage("nonexistent", True)  # should not raise

    def test_refine_steps(self, tmp_path):
        pm = ProceduralMemory(str(tmp_path / "proc.json"))
        p = ProceduralEntry(
            task_type=TaskType.CODE_GENERATION,
            name="code_proc",
            steps=["plan", "code", "test"],
        )
        pm.add(p)
        pm.refine_steps(p.entry_id, ["plan", "design", "code", "test", "review"])
        refined = pm.get(p.entry_id)
        assert len(refined.steps) == 5
        assert "review" in refined.steps

    def test_refine_steps_nonexistent(self, tmp_path):
        pm = ProceduralMemory(str(tmp_path / "proc.json"))
        pm.refine_steps("nonexistent", ["a", "b"])  # should not raise

    def test_search_by_name(self, tmp_path):
        pm = ProceduralMemory(str(tmp_path / "proc.json"))
        pm.add(ProceduralEntry(name="Quick QA"))
        pm.add(ProceduralEntry(name="Deep Analysis"))
        pm.add(ProceduralEntry(name="Quick Summary"))
        found = pm.search_by_name("quick")
        assert len(found) == 2

    def test_top_procedures_requires_min_usage(self, tmp_path):
        pm = ProceduralMemory(str(tmp_path / "proc.json"))
        pm.add(ProceduralEntry(name="new", success_rate=1.0, usage_count=1))
        pm.add(ProceduralEntry(name="proven", success_rate=0.9, usage_count=5))
        pm.add(ProceduralEntry(name="old", success_rate=0.7, usage_count=10))
        top = pm.top_procedures(limit=5)
        assert len(top) == 2  # "new" excluded (usage_count < 2)
        assert top[0].name == "proven"  # highest success_rate

    def test_persistence_roundtrip(self, tmp_path):
        path = str(tmp_path / "proc.json")
        pm1 = ProceduralMemory(path)
        p = ProceduralEntry(
            task_type=TaskType.ANALYSIS,
            name="analysis_proc",
            description="Step by step analysis",
            steps=["gather", "analyse", "conclude"],
            success_rate=0.85,
            usage_count=12,
        )
        pm1.add(p)
        pm2 = ProceduralMemory(path)
        assert pm2.count == 1
        loaded = pm2.get(p.entry_id)
        assert loaded.name == "analysis_proc"
        assert loaded.steps == ["gather", "analyse", "conclude"]
        assert loaded.success_rate == 0.85


# =====================================================================
# MemoryManager
# =====================================================================
class TestMemoryManagerEdge:
    """Edge cases for the unified MemoryManager."""

    def test_recall_with_keyword(self, tmp_path):
        mm = MemoryManager(str(tmp_path))
        mm.store_knowledge(SemanticEntry(
            domain="physics",
            knowledge="Quantum entanglement is non-local",
        ))
        mm.store_knowledge(SemanticEntry(
            domain="math",
            knowledge="Calculus is fundamental",
        ))
        result = mm.recall(TaskType.QUESTION_ANSWERING, [Modality.TEXT], keyword="quantum")
        assert len(result["relevant_knowledge"]) == 1
        assert "entanglement" in result["relevant_knowledge"][0].knowledge

    def test_recall_without_keyword_uses_domain(self, tmp_path):
        mm = MemoryManager(str(tmp_path))
        mm.store_knowledge(SemanticEntry(
            domain="question_answering",
            knowledge="Be concise",
        ))
        result = mm.recall(TaskType.QUESTION_ANSWERING, [Modality.TEXT])
        assert len(result["relevant_knowledge"]) >= 1

    def test_improvement_trend_insufficient_data(self, tmp_path):
        mm = MemoryManager(str(tmp_path))
        trend = mm.improvement_trend(window=10)
        assert trend["trend"] == "insufficient_data"

    def test_improvement_trend_improving(self, tmp_path):
        mm = MemoryManager(str(tmp_path))
        # Add 15 episodes: first 5 low, last 10 high
        for i in range(15):
            score = 4 if i < 5 else 9
            mm.store_episode(EpisodicRecord(
                task_type=TaskType.QUESTION_ANSWERING,
                outcome_score=score,
            ))
        trend = mm.improvement_trend(window=10)
        assert trend["trend"] == "improving"

    def test_improvement_trend_declining(self, tmp_path):
        mm = MemoryManager(str(tmp_path))
        # Add 15 episodes: first 5 high, last 10 low
        for i in range(15):
            score = 9 if i < 5 else 4
            mm.store_episode(EpisodicRecord(
                task_type=TaskType.QUESTION_ANSWERING,
                outcome_score=score,
            ))
        trend = mm.improvement_trend(window=10)
        assert trend["trend"] == "declining"

    def test_improvement_trend_stable(self, tmp_path):
        mm = MemoryManager(str(tmp_path))
        for _ in range(15):
            mm.store_episode(EpisodicRecord(
                task_type=TaskType.QUESTION_ANSWERING,
                outcome_score=7,
            ))
        trend = mm.improvement_trend(window=10)
        assert trend["trend"] == "stable"

    def test_summary_populated(self, tmp_path):
        mm = MemoryManager(str(tmp_path))
        mm.store_episode(EpisodicRecord(outcome_score=8))
        mm.store_knowledge(SemanticEntry(domain="test", knowledge="k"))
        mm.store_procedure(ProceduralEntry(name="p", usage_count=3, success_rate=0.9))
        s = mm.summary()
        assert s["episodic_count"] == 1
        assert s["semantic_count"] == 1
        assert s["procedural_count"] == 1
        assert s["avg_score"] == 8.0

    def test_record_procedure_usage_through_manager(self, tmp_path):
        mm = MemoryManager(str(tmp_path))
        p = ProceduralEntry(
            task_type=TaskType.QUESTION_ANSWERING,
            name="qa",
            success_rate=0.8,
            usage_count=4,
        )
        mm.store_procedure(p)
        mm.record_procedure_usage(p.entry_id, True)
        updated = mm.procedural.get(p.entry_id)
        assert updated.usage_count == 5

    def test_store_episode_triggers_synthesis_check(self, tmp_path):
        mm = MemoryManager(str(tmp_path))
        for i in range(12):
            mm.store_episode(EpisodicRecord(
                task_type=TaskType.QUESTION_ANSWERING,
                user_intent=f"q{i}",
            ))
        # With 12 QA episodes, synthesis threshold (10) is reached
        assert mm.semantic.needs_synthesis("question_answering", 12) is True

    def test_summary_with_empty_memory(self, tmp_path):
        mm = MemoryManager(str(tmp_path))
        s = mm.summary()
        assert s["episodic_count"] == 0
        assert s["semantic_count"] == 0
        assert s["procedural_count"] == 0
        assert s["avg_score"] == 0.0
        assert s["top_procedures"] == []
