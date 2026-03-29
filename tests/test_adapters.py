"""Tests for the trace adapter system."""

import json
import pytest

from agentdiag.adapters import auto_parse
from agentdiag.adapters.generic import GenericAdapter
from agentdiag.adapters.claude import ClaudeAdapter
from agentdiag.adapters.openai import OpenAIAdapter
from agentdiag.adapters.langchain import LangChainAdapter
from agentdiag.loading import load_trace


# -- Fixtures --

GENERIC_TRACE = [
    {"step": 1, "type": "tool_call", "tool": "search", "latency_ms": 500,
     "success": True, "tokens_in": 100, "tokens_out": 200},
    {"step": 2, "type": "tool_call", "tool": "read", "latency_ms": 300,
     "success": True, "tokens_in": 100, "tokens_out": 200},
]

CLAUDE_TRACE = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "Find the bug in my code."}],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Let me analyze the code structure."},
            {"type": "text", "text": "I'll look at your code."},
            {"type": "tool_use", "id": "tu_1", "name": "read_file",
             "input": {"path": "/src/main.py"}},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "tu_1",
             "content": "def main():\n    print('hello')"},
        ],
    },
]

OPENAI_TRACE = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Search for Python docs."},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "web_search", "arguments": '{"q": "python docs"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "Python documentation found."},
    {"role": "assistant", "content": "Here are the Python docs."},
]

LANGCHAIN_TRACE = [
    {
        "run_type": "chain",
        "name": "AgentExecutor",
        "inputs": {"input": "Summarize this document."},
        "outputs": {"output": "planning step"},
        "error": None,
        "latency_ms": 200,
    },
    {
        "run_type": "llm",
        "name": "ChatOpenAI",
        "inputs": {"messages": "..."},
        "outputs": {"text": "I will summarize the document."},
        "error": None,
        "latency_ms": 1500,
    },
    {
        "run_type": "tool",
        "name": "read_document",
        "inputs": {"path": "/doc.txt"},
        "outputs": {"content": "Document text here."},
        "error": None,
        "latency_ms": 300,
    },
]


# -- Auto-detection tests --

class TestAutoDetection:
    def test_detects_generic(self):
        adapter = GenericAdapter()
        assert adapter.can_parse(GENERIC_TRACE)
        assert not adapter.can_parse(CLAUDE_TRACE)
        assert not adapter.can_parse(OPENAI_TRACE)

    def test_detects_claude(self):
        adapter = ClaudeAdapter()
        assert adapter.can_parse(CLAUDE_TRACE)
        assert not adapter.can_parse(GENERIC_TRACE)
        assert not adapter.can_parse(OPENAI_TRACE)

    def test_detects_openai(self):
        adapter = OpenAIAdapter()
        assert adapter.can_parse(OPENAI_TRACE)
        assert not adapter.can_parse(CLAUDE_TRACE)

    def test_detects_langchain(self):
        adapter = LangChainAdapter()
        assert adapter.can_parse(LANGCHAIN_TRACE)
        assert not adapter.can_parse(GENERIC_TRACE)


# -- Parsing tests --

class TestGenericAdapter:
    def test_parse(self):
        events = GenericAdapter().parse(GENERIC_TRACE)
        assert len(events) == 2
        assert events[0].tool == "search"
        assert events[1].tool == "read"


class TestClaudeAdapter:
    def test_parse(self):
        events = ClaudeAdapter().parse(CLAUDE_TRACE)
        assert len(events) > 0
        # Should have: planning (thinking), reasoning (text), tool_call (tool_use), tool_call (tool_result)
        types = [e.type for e in events]
        assert "planning" in types
        assert "tool_call" in types

    def test_tool_use_parsed(self):
        events = ClaudeAdapter().parse(CLAUDE_TRACE)
        tool_calls = [e for e in events if e.type == "tool_call" and e.tool == "read_file"]
        assert len(tool_calls) == 1


class TestOpenAIAdapter:
    def test_parse(self):
        events = OpenAIAdapter().parse(OPENAI_TRACE)
        assert len(events) > 0
        tool_calls = [e for e in events if e.type == "tool_call"]
        assert len(tool_calls) >= 1

    def test_reasoning_parsed(self):
        events = OpenAIAdapter().parse(OPENAI_TRACE)
        reasoning = [e for e in events if e.type == "reasoning"]
        assert len(reasoning) >= 1


class TestLangChainAdapter:
    def test_parse(self):
        events = LangChainAdapter().parse(LANGCHAIN_TRACE)
        assert len(events) == 3
        types = [e.type for e in events]
        assert "planning" in types   # chain
        assert "reasoning" in types  # llm
        assert "tool_call" in types  # tool

    def test_latency_preserved(self):
        events = LangChainAdapter().parse(LANGCHAIN_TRACE)
        tool_event = [e for e in events if e.type == "tool_call"][0]
        assert tool_event.latency_ms == 300


# -- auto_parse integration --

class TestAutoParse:
    def test_auto_parses_generic(self):
        events = auto_parse(GENERIC_TRACE)
        assert len(events) == 2

    def test_auto_parses_claude(self):
        events = auto_parse(CLAUDE_TRACE)
        assert len(events) > 0

    def test_auto_parses_openai(self):
        events = auto_parse(OPENAI_TRACE)
        assert len(events) > 0

    def test_auto_parses_langchain(self):
        events = auto_parse(LANGCHAIN_TRACE)
        assert len(events) == 3

    def test_raises_on_unknown(self):
        with pytest.raises(ValueError, match="No adapter matched"):
            auto_parse([{"unknown_key": "data"}])


# -- load_trace integration --

class TestLoadTraceWithAdapters:
    def test_load_claude_trace(self, tmp_path):
        path = tmp_path / "claude_trace.json"
        path.write_text(json.dumps(CLAUDE_TRACE))
        events = load_trace(path)
        assert len(events) > 0

    def test_load_openai_trace(self, tmp_path):
        path = tmp_path / "openai_trace.json"
        path.write_text(json.dumps(OPENAI_TRACE))
        events = load_trace(path)
        assert len(events) > 0
