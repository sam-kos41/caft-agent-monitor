"""LLM-powered explanation engine (commercial tier, stubbed)."""

from __future__ import annotations

from agentdiag.models import TraceEvent, Diagnosis
from agentdiag.explain.templates import ExplanationResult


class LLMExplainer:
    """Uses an LLM to generate context-aware explanations.

    This is a commercial-tier feature. The free tier uses
    template-based explanations via agentdiag.explain.templates.
    """

    def explain(
        self,
        diagnosis: Diagnosis,
        events: list[TraceEvent],
    ) -> ExplanationResult:
        """Generate a context-aware explanation using an LLM.

        Args:
            diagnosis: The failure diagnosis to explain.
            events: The full trace events for context.

        Raises:
            NotImplementedError: This feature requires agentdiag-pro.
        """
        raise NotImplementedError("LLM explanations require agentdiag-pro")
