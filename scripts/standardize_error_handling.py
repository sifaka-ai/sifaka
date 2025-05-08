#!/usr/bin/env python3
"""
Script to standardize error handling in Sifaka components.

This script:
1. Finds all component files (classifiers, critics, models, rules, chains)
2. Checks for consistent error handling patterns
3. Reports components that need to be updated

Usage:
    python standardize_error_handling.py [--fix] [--verbose]
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

# Error handling patterns to check for
ERROR_PATTERNS = {
    "try_except": r"try\s*:",  # Check for try-except blocks
    "except_block": r"except\s+(?:Exception|[A-Za-z]+Error)",  # Check for except blocks
    "error_logging": r"logger\.(?:error|warning|exception)",  # Check for error logging
    "error_metadata": r"metadata\s*=\s*\{[^}]*\"error\"",  # Check for error metadata
    "error_propagation": r"raise\s+(?:RuntimeError|ValueError|TypeError|SifakaError|ModelError|ClassifierError|CriticError|ChainError)",  # Check for error propagation
    "sifaka_error_import": r"from\s+sifaka\.utils(?:\.errors)?\s+import\s+(?:SifakaError|ValidationError|ConfigurationError|ModelError|ClassifierError|CriticError|ChainError)",  # Check for Sifaka error imports
    "error_handling_utils": r"(?:format_error_metadata|handle_errors|with_error_handling)",  # Check for error handling utilities
}

# Component types and their directories
COMPONENT_DIRS = {
    "classifiers": "sifaka/classifiers",
    "critics": "sifaka/critics",
    "models": "sifaka/models",
    "rules": "sifaka/rules",
    "chains": "sifaka/chain",
}

# Standard error handling template for different component types
ERROR_TEMPLATES = {
    "classifiers": """
        from sifaka.utils import ClassifierError, format_error_metadata, get_logger

        logger = get_logger(__name__)

        try:
            # Core classification logic
            # ...
            return ClassificationResult(
                label="some_label",
                confidence=0.8,
                metadata={"processed_successfully": True}
            )
        except Exception as e:
            # Log the error
            logger.error(f"Classification error: {e}")

            # Return a fallback result with standardized error metadata
            return ClassificationResult(
                label="unknown",
                confidence=0.0,
                metadata=format_error_metadata(e)
            )
    """,
    "critics": """
        from sifaka.utils import CriticError, format_error_metadata, get_logger

        logger = get_logger(__name__)

        try:
            # Core critique logic
            # ...
            return CriticMetadata(
                score=0.8,
                feedback="Good quality text",
                issues=[],
                suggestions=[]
            )
        except Exception as e:
            # Log the error
            logger.error(f"Critique error: {e}")

            # Return a fallback result with standardized error metadata
            return CriticMetadata(
                score=0.0,
                feedback=f"Error during critique: {str(e)}",
                issues=["Critique process failed"],
                suggestions=[],
                metadata=format_error_metadata(e)
            )
    """,
    "models": """
        from sifaka.utils import ModelError, with_error_handling, get_logger

        logger = get_logger(__name__)

        try:
            # Core generation logic
            # ...
            return response_text
        except Exception as e:
            # Log the error
            logger.error(f"Generation error: {e}")

            # Re-raise with context using the specific error type
            raise ModelError(f"Error generating text: {e}", cause=e)
    """,
    "rules": """
        from sifaka.utils import ValidationError, format_error_metadata, get_logger

        logger = get_logger(__name__)

        try:
            # Core validation logic
            # ...
            return RuleResult(
                passed=True,
                message="Validation passed",
                metadata={}
            )
        except Exception as e:
            # Log the error
            logger.error(f"Validation error: {e}")

            # Return a failure result with standardized error metadata
            return RuleResult(
                passed=False,
                message=f"Error during validation: {str(e)}",
                metadata=format_error_metadata(e)
            )
    """,
    "chains": """
        from sifaka.utils import ChainError, with_error_handling, get_logger

        logger = get_logger(__name__)

        # Using the context manager for error handling
        with with_error_handling("chain execution", logger=logger):
            # Core chain logic
            # ...
            return result

        # Alternative approach with try-except
        try:
            # Core chain logic
            # ...
            return result
        except Exception as e:
            # Log the error
            logger.error(f"Chain error: {e}")

            # Re-raise with context using the specific error type
            raise ChainError(f"Error running chain: {e}", cause=e)
    """,
}


def find_component_files(component_type: str) -> List[str]:
    """
    Find all files for a specific component type.

    Args:
        component_type: Type of component to find files for

    Returns:
        List of file paths
    """
    component_files = []

    search_dir = COMPONENT_DIRS.get(component_type)
    if not search_dir:
        print(f"Unknown component type: {component_type}")
        return []

    for root, _, files in os.walk(search_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                component_files.append(file_path)

    return component_files


def check_error_handling(file_path: str) -> Dict[str, bool]:
    """
    Check if a file has consistent error handling.

    Args:
        file_path: Path to the file to check

    Returns:
        Dictionary of error handling patterns and whether they are present
    """
    results = {pattern: False for pattern in ERROR_PATTERNS}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for each error handling pattern
        for pattern, regex in ERROR_PATTERNS.items():
            if re.search(regex, content):
                results[pattern] = True
    except Exception as e:
        print(f"Error checking error handling in {file_path}: {str(e)}")

    return results


def analyze_component_files(
    component_type: str, verbose: bool = False
) -> Dict[str, Dict[str, bool]]:
    """
    Analyze all files for a specific component type.

    Args:
        component_type: Type of component to analyze
        verbose: If True, print more information

    Returns:
        Dictionary of file paths and their error handling patterns
    """
    if verbose:
        print(f"Analyzing {component_type} files...")

    component_files = find_component_files(component_type)
    results = {}

    for file_path in component_files:
        if verbose:
            print(f"  Checking {file_path}...")

        results[file_path] = check_error_handling(file_path)

    return results


def report_issues(
    analysis: Dict[str, Dict[str, Dict[str, bool]]], verbose: bool = False
) -> Dict[str, List[str]]:
    """
    Report issues with error handling.

    Args:
        analysis: Analysis of error handling patterns
        verbose: If True, print more information

    Returns:
        Dictionary of component types and files that need to be updated
    """
    issues = {}

    for component_type, files in analysis.items():
        component_issues = []

        for file_path, patterns in files.items():
            # Check if all patterns are present
            if not all(patterns.values()):
                component_issues.append(file_path)

                if verbose:
                    missing_patterns = [
                        pattern for pattern, present in patterns.items() if not present
                    ]
                    print(f"  {file_path} is missing: {', '.join(missing_patterns)}")

        issues[component_type] = component_issues

    return issues


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Standardize error handling in Sifaka components")
    parser.add_argument("--fix", action="store_true", help="Fix issues (not implemented yet)")
    parser.add_argument("--verbose", action="store_true", help="Print more information")
    args = parser.parse_args()

    # Analyze all component types
    analysis = {}
    for component_type in COMPONENT_DIRS:
        analysis[component_type] = analyze_component_files(component_type, args.verbose)

    # Report issues
    issues = report_issues(analysis, args.verbose)

    # Print summary
    total_files = sum(len(files) for files in analysis.values())
    total_issues = sum(len(component_issues) for component_issues in issues.values())

    print(f"\nAnalyzed {total_files} files, found {total_issues} with error handling issues")

    for component_type, component_issues in issues.items():
        if component_issues:
            print(f"  {component_type}: {len(component_issues)} files need updating")

    # Print example error handling templates
    if args.verbose:
        print("\nExample error handling templates:")
        for component_type, template in ERROR_TEMPLATES.items():
            print(f"\n{component_type.upper()} TEMPLATE:")
            print(template)

    return 0


if __name__ == "__main__":
    sys.exit(main())
