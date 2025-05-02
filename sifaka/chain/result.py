"""
Chain result module for Sifaka.

This module provides the ChainResult class which represents the result of running a chain.
"""

from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar, Dict, Any

from ..rules import RuleResult

OutputType = TypeVar("OutputType")


@dataclass
class ChainResult(Generic[OutputType]):
    """Result from running a chain, including the output and validation details."""

    output: OutputType
    rule_results: List[RuleResult]
    critique_details: Optional[Dict[str, Any]] = None
