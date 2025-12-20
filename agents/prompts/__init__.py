# agents/prompts/__init__.py
"""LLM prompt templates for agent tool usage."""

from .system_prompt import SYSTEM_PROMPT, get_tool_descriptions
from .few_shot_examples import FEW_SHOT_EXAMPLES, get_examples_for_tool

__all__ = [
    "SYSTEM_PROMPT",
    "get_tool_descriptions",
    "FEW_SHOT_EXAMPLES",
    "get_examples_for_tool",
]
