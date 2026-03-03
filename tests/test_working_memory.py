"""Tests for WorkingMemory — volatile session-scoped context buffer."""

import pytest

from pinocchio.memory.working_memory import WorkingMemory, WorkingMemoryItem


class TestWorkingMemoryItem:
    """WorkingMemoryItem dataclass basics."""

    def test_defaults(self):
        item = WorkingMemoryItem()
        assert item.item_id  # auto-generated non-empty string
        assert item.category == ""
        assert item.content == ""
        assert item.source == ""
        assert item.relevance == 1.0
        assert item.created_at  # auto-populated
        assert item.metadata == {}

    def test_to_dict(self):
        item = WorkingMemoryItem(
            category="conversation",
            content="hello",
            source="user",
            metadata={"turn": 1},
        )
        d = item.to_dict()
        assert d["category"] == "conversation"
        assert d["content"] == "hello"
        assert d["source"] == "user"
        assert d["metadata"] == {"turn": 1}
        assert "item_id" in d
        assert "created_at" in d

    def test_custom_relevance(self):
        item = WorkingMemoryItem(relevance=0.5)
        assert item.relevance == 0.5


class TestWorkingMemoryCapacity:
    """Capacity limit and FIFO eviction."""

    def test_default_capacity(self):
        wm = WorkingMemory()
        assert wm.capacity == 50
        assert wm.count == 0

    def test_custom_capacity(self):
        wm = WorkingMemory(capacity=5)
        assert wm.capacity == 5

    def test_add_within_capacity(self):
        wm = WorkingMemory(capacity=3)
        wm.add(WorkingMemoryItem(content="a"))
        wm.add(WorkingMemoryItem(content="b"))
        assert wm.count == 2

    def test_eviction_at_capacity(self):
        wm = WorkingMemory(capacity=3)
        wm.add(WorkingMemoryItem(content="a"))
        wm.add(WorkingMemoryItem(content="b"))
        wm.add(WorkingMemoryItem(content="c"))
        wm.add(WorkingMemoryItem(content="d"))
        assert wm.count == 3
        contents = [it.content for it in wm.all_items()]
        assert contents == ["b", "c", "d"]  # "a" was evicted

    def test_clear(self):
        wm = WorkingMemory(capacity=10)
        wm.add_conversation_turn("user", "hello")
        wm.add_conversation_turn("assistant", "hi")
        wm.clear()
        assert wm.count == 0
        assert wm.turn_count == 0


class TestConversationTurns:
    """Conversation turn management."""

    def test_add_conversation_turn(self):
        wm = WorkingMemory()
        item = wm.add_conversation_turn("user", "hello")
        assert item.category == "conversation"
        assert item.source == "user"
        assert item.content == "hello"
        assert wm.turn_count == 1

    def test_multiple_turns(self):
        wm = WorkingMemory()
        wm.add_conversation_turn("user", "hello")
        wm.add_conversation_turn("assistant", "hi there")
        wm.add_conversation_turn("user", "how are you?")
        assert wm.turn_count == 3
        assert wm.count == 3

    def test_get_conversation(self):
        wm = WorkingMemory()
        wm.add_conversation_turn("user", "hello")
        wm.add_hypothesis("maybe X")
        wm.add_conversation_turn("assistant", "hi")
        conv = wm.get_conversation()
        assert len(conv) == 2
        assert conv[0].content == "hello"
        assert conv[1].content == "hi"

    def test_metadata_in_conversation_turn(self):
        wm = WorkingMemory()
        item = wm.add_conversation_turn("user", "hello", modality="text")
        assert item.metadata == {"modality": "text"}


class TestHypothesesAndContext:
    """Hypothesis and context item management."""

    def test_add_hypothesis(self):
        wm = WorkingMemory()
        item = wm.add_hypothesis("The user wants code help", source="strategy")
        assert item.category == "hypothesis"
        assert item.source == "strategy"
        assert item.content == "The user wants code help"
        assert item.relevance == 1.0

    def test_add_context(self):
        wm = WorkingMemory()
        item = wm.add_context("Recalled: user prefers Python", source="recall")
        assert item.category == "context"
        assert item.source == "recall"
        assert item.relevance == 0.8  # context starts lower

    def test_get_hypotheses(self):
        wm = WorkingMemory()
        wm.add_hypothesis("hyp1")
        wm.add_hypothesis("hyp2")
        wm.add_context("ctx1")
        hyps = wm.get_hypotheses()
        assert len(hyps) == 2


class TestSearch:
    """Keyword search across items."""

    def test_basic_search(self):
        wm = WorkingMemory()
        wm.add_conversation_turn("user", "help with Python code")
        wm.add_conversation_turn("user", "what about Java?")
        wm.add_hypothesis("Python is preferred")
        results = wm.search("python")
        assert len(results) == 2

    def test_case_insensitive(self):
        wm = WorkingMemory()
        wm.add_conversation_turn("user", "Hello World")
        results = wm.search("hello")
        assert len(results) == 1

    def test_search_limit(self):
        wm = WorkingMemory()
        for i in range(20):
            wm.add_conversation_turn("user", f"apple message {i}")
        results = wm.search("apple", limit=5)
        assert len(results) == 5

    def test_search_no_match(self):
        wm = WorkingMemory()
        wm.add_conversation_turn("user", "hello")
        assert wm.search("xyz") == []


class TestRelevanceDecay:
    """Relevance decay mechanics."""

    def test_decay_on_conversation_turn(self):
        wm = WorkingMemory()
        ctx = wm.add_context("some context")
        assert ctx.relevance == 0.8
        wm.add_conversation_turn("user", "turn 1")
        # After a turn, non-conversation items decay
        assert ctx.relevance < 0.8

    def test_conversation_items_not_decayed(self):
        wm = WorkingMemory()
        turn = wm.add_conversation_turn("user", "turn 1")
        # Store relevance
        initial_relevance = turn.relevance
        wm.add_conversation_turn("user", "turn 2")
        # Conversation items should NOT decay
        assert turn.relevance == initial_relevance

    def test_get_relevant_threshold(self):
        wm = WorkingMemory()
        wm.add(WorkingMemoryItem(category="context", content="high", relevance=0.9))
        wm.add(WorkingMemoryItem(category="context", content="low", relevance=0.1))
        wm.add(WorkingMemoryItem(category="context", content="mid", relevance=0.5))
        relevant = wm.get_relevant(threshold=0.4)
        assert len(relevant) == 2
        assert relevant[0].content == "high"
        assert relevant[1].content == "mid"

    def test_relevance_never_below_zero(self):
        wm = WorkingMemory()
        ctx = wm.add_context("test")
        # Add many turns to force heavy decay
        for i in range(100):
            wm.add_conversation_turn("user", f"turn {i}")
        assert ctx.relevance >= 0.0


class TestContextFormatting:
    """Context formatting for LLM prompts."""

    def test_format_conversation_context(self):
        wm = WorkingMemory()
        wm.add_conversation_turn("user", "What is Python?")
        wm.add_conversation_turn("assistant", "Python is a programming language.")
        fmt = wm.format_conversation_context()
        assert "User: What is Python?" in fmt
        assert "Assistant: Python is a programming language." in fmt

    def test_format_conversation_empty(self):
        wm = WorkingMemory()
        assert wm.format_conversation_context() == ""

    def test_format_conversation_max_turns(self):
        wm = WorkingMemory()
        for i in range(20):
            wm.add_conversation_turn("user", f"turn {i}")
        fmt = wm.format_conversation_context(max_turns=5)
        lines = fmt.strip().split("\n")
        assert len(lines) == 5
        assert "turn 15" in lines[0]  # oldest of the last 5

    def test_format_active_context(self):
        wm = WorkingMemory()
        wm.add_hypothesis("maybe code help")
        wm.add_context("user prefers Python")
        wm.add_conversation_turn("user", "help me")  # should be excluded
        fmt = wm.format_active_context()
        assert "hypothesis" in fmt
        assert "context" in fmt
        assert "help me" not in fmt

    def test_format_active_context_empty(self):
        wm = WorkingMemory()
        wm.add_conversation_turn("user", "hi")
        assert wm.format_active_context() == ""


class TestSummary:
    """Summary output for status/debug."""

    def test_summary_empty(self):
        wm = WorkingMemory()
        s = wm.summary()
        assert s["total_items"] == 0
        assert s["capacity"] == 50
        assert s["turn_count"] == 0
        assert s["categories"] == {}

    def test_summary_with_items(self):
        wm = WorkingMemory()
        wm.add_conversation_turn("user", "hi")
        wm.add_conversation_turn("assistant", "hello")
        wm.add_hypothesis("test hypothesis")
        s = wm.summary()
        assert s["total_items"] == 3
        assert s["turn_count"] == 2
        assert s["categories"]["conversation"] == 2
        assert s["categories"]["hypothesis"] == 1
