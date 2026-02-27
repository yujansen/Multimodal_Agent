"""Tests for data models and enumerations."""

from __future__ import annotations

import pytest

from pinocchio.models.enums import (
    Modality,
    TaskType,
    Complexity,
    ConfidenceLevel,
    ErrorType,
    FusionStrategy,
    AgentRole,
    ExpertiseLevel,
    CommunicationStyle,
)
from pinocchio.models.schemas import (
    EpisodicRecord,
    SemanticEntry,
    ProceduralEntry,
    PerceptionResult,
    StrategyResult,
    EvaluationResult,
    LearningResult,
    MetaReflectionResult,
    UserModel,
    ModalConfidence,
    MultimodalInput,
    AgentMessage,
)


# ──────────────────────────────────────────────────────────────────
# Enum Tests
# ──────────────────────────────────────────────────────────────────


class TestEnums:
    def test_modality_values(self):
        assert Modality.TEXT.value == "text"
        assert Modality.IMAGE.value == "image"
        assert Modality.AUDIO.value == "audio"
        assert Modality.VIDEO.value == "video"

    def test_task_type_has_unknown(self):
        assert TaskType.UNKNOWN.value == "unknown"
        assert len(TaskType) == 11

    def test_complexity_is_integer(self):
        assert Complexity.TRIVIAL == 1
        assert Complexity.EXTREME == 5

    def test_error_type_values(self):
        assert ErrorType.PERCEPTION_ERROR.value == "perception_error"
        assert ErrorType.CROSS_MODAL_ERROR.value == "cross_modal_error"

    def test_agent_roles_complete(self):
        roles = [r.value for r in AgentRole]
        assert "orchestrator" in roles
        assert "perception" in roles
        assert "meta_reflection" in roles
        assert "vision_processor" in roles


# ──────────────────────────────────────────────────────────────────
# Schema Tests
# ──────────────────────────────────────────────────────────────────


class TestEpisodicRecord:
    def test_defaults(self):
        ep = EpisodicRecord()
        assert ep.episode_id  # auto-generated
        assert ep.timestamp   # auto-generated
        assert ep.task_type == TaskType.UNKNOWN
        assert ep.modalities == []
        assert ep.outcome_score == 5

    def test_to_dict_and_from_dict_roundtrip(self):
        ep = EpisodicRecord(
            task_type=TaskType.CODE_GENERATION,
            modalities=[Modality.TEXT, Modality.IMAGE],
            user_intent="Write a sorting function",
            strategy_used="code_gen_v2",
            outcome_score=8,
            lessons=["Use type hints", "Add docstrings"],
            error_patterns=[],
            improvement_notes="Consider edge cases",
        )
        d = ep.to_dict()
        restored = EpisodicRecord.from_dict(d)

        assert restored.task_type == TaskType.CODE_GENERATION
        assert restored.modalities == [Modality.TEXT, Modality.IMAGE]
        assert restored.outcome_score == 8
        assert restored.lessons == ["Use type hints", "Add docstrings"]
        assert restored.improvement_notes == "Consider edge cases"

    def test_from_dict_with_missing_fields(self):
        ep = EpisodicRecord.from_dict({"task_type": "analysis"})
        assert ep.task_type == TaskType.ANALYSIS
        assert ep.modalities == []
        assert ep.outcome_score == 5


class TestSemanticEntry:
    def test_defaults(self):
        entry = SemanticEntry()
        assert entry.confidence == 0.5
        assert entry.source_episodes == []

    def test_to_dict(self):
        entry = SemanticEntry(domain="coding", knowledge="Use descriptive names")
        d = entry.to_dict()
        assert d["domain"] == "coding"
        assert d["knowledge"] == "Use descriptive names"


class TestProceduralEntry:
    def test_defaults(self):
        entry = ProceduralEntry()
        assert entry.success_rate == 0.0
        assert entry.usage_count == 0

    def test_to_dict_roundtrip(self):
        entry = ProceduralEntry(
            task_type=TaskType.SUMMARIZATION,
            name="summarize_v1",
            steps=["Read document", "Extract key points", "Write summary"],
            success_rate=0.85,
            usage_count=10,
        )
        d = entry.to_dict()
        restored = ProceduralEntry.from_dict(d)
        assert restored.name == "summarize_v1"
        assert restored.task_type == TaskType.SUMMARIZATION
        assert len(restored.steps) == 3
        assert restored.success_rate == 0.85


class TestMultimodalInput:
    def test_text_only(self):
        inp = MultimodalInput(text="Hello")
        assert inp.modalities == [Modality.TEXT]

    def test_multimodal(self):
        inp = MultimodalInput(
            text="Describe this",
            image_paths=["img.jpg"],
            audio_paths=["aud.wav"],
        )
        mods = inp.modalities
        assert Modality.TEXT in mods
        assert Modality.IMAGE in mods
        assert Modality.AUDIO in mods
        assert Modality.VIDEO not in mods

    def test_no_modalities(self):
        inp = MultimodalInput()
        assert inp.modalities == []


class TestModalConfidence:
    def test_to_dict(self):
        mc = ModalConfidence(text=0.9, image=0.7)
        d = mc.to_dict()
        assert d["text"] == 0.9
        assert d["image"] == 0.7
        assert d["audio"] == 0.0


class TestAgentMessage:
    def test_defaults(self):
        msg = AgentMessage(content="Hello")
        assert msg.role == "assistant"
        assert msg.modality == Modality.TEXT
        assert msg.confidence == 1.0


class TestUserModel:
    def test_defaults(self):
        user = UserModel()
        assert user.expertise == ExpertiseLevel.INTERMEDIATE
        assert user.interaction_count == 0


class TestCognitiveResultSchemas:
    """Ensure all cognitive loop result dataclasses instantiate correctly."""

    def test_perception_result(self):
        r = PerceptionResult(task_type=TaskType.ANALYSIS, complexity=Complexity.COMPLEX)
        assert r.task_type == TaskType.ANALYSIS
        assert r.complexity == Complexity.COMPLEX

    def test_strategy_result(self):
        r = StrategyResult(selected_strategy="test_strat", is_novel=True)
        assert r.is_novel is True

    def test_evaluation_result(self):
        r = EvaluationResult(output_quality=9, went_well=["accuracy"])
        assert r.output_quality == 9

    def test_learning_result(self):
        r = LearningResult(new_lessons=["lesson1"], skill_gap="math")
        assert r.skill_gap == "math"

    def test_meta_reflection_result(self):
        r = MetaReflectionResult(strength_domains=["coding"], weakness_domains=["art"])
        assert "coding" in r.strength_domains
