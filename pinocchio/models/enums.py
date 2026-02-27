"""Enumerations for the Pinocchio agent system."""

from enum import Enum


class Modality(str, Enum):
    """Supported input/output modalities."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class TaskType(str, Enum):
    """Classification of task types."""

    QUESTION_ANSWERING = "question_answering"
    CONTENT_GENERATION = "content_generation"
    ANALYSIS = "analysis"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    CODE_GENERATION = "code_generation"
    CREATIVE_WRITING = "creative_writing"
    MULTIMODAL_REASONING = "multimodal_reasoning"
    CONVERSATION = "conversation"
    TOOL_USE = "tool_use"
    UNKNOWN = "unknown"


class Complexity(int, Enum):
    """Task complexity levels (1-5)."""

    TRIVIAL = 1
    SIMPLE = 2
    MODERATE = 3
    COMPLEX = 4
    EXTREME = 5


class ConfidenceLevel(str, Enum):
    """Confidence level for agent decisions."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ErrorType(str, Enum):
    """Classification of error types for root cause analysis."""

    PERCEPTION_ERROR = "perception_error"        # Misunderstood input/intent
    STRATEGY_ERROR = "strategy_error"            # Chose wrong approach
    EXECUTION_ERROR = "execution_error"          # Correct strategy, poor impl
    KNOWLEDGE_GAP = "knowledge_gap"              # Lacked necessary information
    CROSS_MODAL_ERROR = "cross_modal_error"      # Inconsistency between modalities


class FusionStrategy(str, Enum):
    """Multimodal fusion strategies."""

    EARLY_FUSION = "early_fusion"    # Combine raw features before reasoning
    LATE_FUSION = "late_fusion"      # Reason separately, then integrate
    HYBRID_FUSION = "hybrid_fusion"  # Mix of early and late


class AgentRole(str, Enum):
    """Roles of sub-agents in the Pinocchio system."""

    ORCHESTRATOR = "orchestrator"
    PERCEPTION = "perception"
    STRATEGY = "strategy"
    EXECUTION = "execution"
    EVALUATION = "evaluation"
    LEARNING = "learning"
    META_REFLECTION = "meta_reflection"
    TEXT_PROCESSOR = "text_processor"
    VISION_PROCESSOR = "vision_processor"
    AUDIO_PROCESSOR = "audio_processor"
    VIDEO_PROCESSOR = "video_processor"


class ExpertiseLevel(str, Enum):
    """User expertise levels for adaptive communication."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"


class CommunicationStyle(str, Enum):
    """User communication style preferences."""

    CONCISE = "concise"
    DETAILED = "detailed"
    VISUAL = "visual"
