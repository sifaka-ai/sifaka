"""Type stubs for sifaka.presets module.

This file provides type hints for preset functions with better IDE support.
"""

from typing import Any, Awaitable
from sifaka.core.thought import SifakaThought

# Academic writing preset
def academic_writing(
    prompt: str,
    *,
    min_length: int = 300,
    max_rounds: int = 5,
    model: str = "openai:gpt-4",
    **kwargs: Any
) -> Awaitable[SifakaThought]: ...

# Creative writing preset
def creative_writing(
    prompt: str,
    *,
    max_length: int = 800,
    max_rounds: int = 4,
    model: str = "anthropic:claude-3-5-sonnet-20241022",
    **kwargs: Any
) -> Awaitable[SifakaThought]: ...

# Technical documentation preset
def technical_docs(
    prompt: str,
    *,
    min_length: int = 200,
    max_rounds: int = 4,
    model: str = "openai:gpt-4",
    **kwargs: Any
) -> Awaitable[SifakaThought]: ...

# Business writing preset
def business_writing(
    prompt: str,
    *,
    max_length: int = 500,
    max_rounds: int = 3,
    model: str = "openai:gpt-4o-mini",
    **kwargs: Any
) -> Awaitable[SifakaThought]: ...

# Quick draft preset
def quick_draft(
    prompt: str,
    *,
    max_rounds: int = 2,
    model: str = "openai:gpt-4o-mini",
    **kwargs: Any
) -> Awaitable[SifakaThought]: ...

# High quality preset
def high_quality(
    prompt: str,
    *,
    min_length: int = 400,
    max_rounds: int = 7,
    model: str = "openai:gpt-4",
    **kwargs: Any
) -> Awaitable[SifakaThought]: ...

# Convenience aliases
academic: type[academic_writing]
creative: type[creative_writing]
technical: type[technical_docs]
business: type[business_writing]
draft: type[quick_draft]
premium: type[high_quality]
