"""Type stubs for sifaka package.

This file provides type hints for better IDE support and static type checking.
"""

from typing import Any, Dict, List, Optional, Union, Awaitable
from sifaka.core.thought import SifakaThought

# Main API functions
def improve(
    prompt: str,
    *,
    max_rounds: int = 3,
    model: str = "openai:gpt-4",
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    critics: Optional[List[str]] = None,
    enable_logging: bool = False,
    enable_timing: bool = False,
    enable_caching: bool = False,
    **kwargs: Any
) -> Awaitable[SifakaThought]: ...

def simple(
    prompt: str,
    *,
    model: str = "openai:gpt-4",
    **kwargs: Any
) -> Awaitable[SifakaThought]: ...

def fluent(prompt: str) -> "FluentAPI": ...

# Fluent API (deprecated but maintained for compatibility)
class FluentAPI:
    def __init__(self, prompt: str) -> None: ...
    def model(self, model: str) -> "FluentAPI": ...
    def rounds(self, max_rounds: int) -> "FluentAPI": ...
    def length(self, min_length: Optional[int] = None, max_length: Optional[int] = None) -> "FluentAPI": ...
    def critics(self, *critics: str) -> "FluentAPI": ...
    def logging(self, enabled: bool = True) -> "FluentAPI": ...
    def timing(self, enabled: bool = True) -> "FluentAPI": ...
    def caching(self, enabled: bool = True) -> "FluentAPI": ...
    def run(self, **kwargs: Any) -> Awaitable[SifakaThought]: ...

# Version information
__version__: str

# Package metadata
__author__: str
__email__: str
__description__: str
