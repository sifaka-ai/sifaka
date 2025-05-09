"""
Chain Result Module

A brief description of the module's purpose and functionality.

## Overview
This module provides the ChainResult class which represents the result of running a chain,
including the generated output, validation results, and optional critique details.

## Components
1. **ChainResult**: Main result class that encapsulates chain execution results
2. **RuleResult**: Individual rule validation results
3. **CritiqueDetails**: Optional details from critic refinement

## Usage Examples
```python
from sifaka.chain import ChainResult
from sifaka.rules import RuleResult

# Create a chain result
result = ChainResult(
    output="Generated text output",
    rule_results=[
        RuleResult(
            rule_name="length_rule",
            passed=True,
            details={"length": 100}
        )
    ],
    critique_details={
        "feedback": "Good text, but could be more concise",
        "suggestions": ["Remove redundant phrases", "Use active voice"]
    }
)

# Access result properties
print(f"Output: {result.output}")
print(f"All rules passed: {all(r.passed for r in result.rule_results)}")
if result.critique_details:
    print(f"Critique feedback: {result.critique_details.get('feedback', '')}")
```

## Error Handling
- No specific exceptions raised by this module

## Configuration
- output: The generated output from the chain
- rule_results: List of validation results from rules
- critique_details: Optional dictionary of critique details
"""

from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar, Dict, Any

from ..rules import RuleResult

OutputType = TypeVar("OutputType")


@dataclass
class ChainResult(Generic[OutputType]):
    """
    Result from running a chain, including the output and validation details.

    Detailed description of the class's purpose, functionality, and usage.

    ## Architecture
    ChainResult follows a simple data structure:
    1. **Core Data**: Essential data
       - Output: Generated output
       - Rule Results: Validation results
       - Critique Details: Optional feedback

    ## Lifecycle
    1. **Creation**: Initialize with data
       - Set output
       - Set rule results
       - Set critique details
    2. **Usage**: Access data
       - Read output
       - Check validation
       - Get feedback

    ## Error Handling
    - No specific exceptions raised by this class

    ## Examples
    ```python
    from sifaka.chain import ChainResult
    from sifaka.rules import RuleResult

    # Create a chain result
    result = ChainResult(
        output="Generated text output",
        rule_results=[
            RuleResult(
                rule_name="length_rule",
                passed=True,
                details={"length": 100}
            )
        ],
        critique_details={
            "feedback": "Good text, but could be more concise",
            "suggestions": ["Remove redundant phrases", "Use active voice"]
        }
    )

    # Access result properties
    print(f"Output: {result.output}")
    print(f"All rules passed: {all(r.passed for r in result.rule_results)}")
    if result.critique_details:
        print(f"Critique feedback: {result.critique_details.get('feedback', '')}")
    ```

    Attributes:
        output (OutputType): The generated output from the chain
        rule_results (List[RuleResult]): List of validation results from rules
        critique_details (Optional[Dict[str, Any]]): Optional dictionary of critique details
    """

    output: OutputType
    rule_results: List[RuleResult]
    critique_details: Optional[Dict[str, Any]] = None
