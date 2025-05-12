#!/usr/bin/env python3
"""
Script to update imports in the Sifaka codebase.

This script updates imports from the old module structure to the new modular directory structure:
1. sifaka.utils.config -> sifaka.utils.config.{base,models,rules,critics,chain,classifiers,retrieval}
2. sifaka.utils.errors -> sifaka.utils.errors.{base,component,handling,results,safe_execution,logging}
3. sifaka.core.dependency -> sifaka.core.dependency.{provider,scopes,injector,utils}

Usage:
    python update_imports.py [--dry-run] [--verbose] [path]

Options:
    --dry-run   Don't write changes, just print what would be changed
    --verbose   Print detailed information about changes
    path        Path to process (default: current directory)
"""

import re
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set

# Define mappings from old imports to new imports
CONFIG_MAPPINGS = {
    # Base config
    r'from sifaka\.utils\.config import BaseConfig': 'from sifaka.utils.config.base import BaseConfig',
    
    # Model configs
    r'from sifaka\.utils\.config import ModelConfig': 'from sifaka.utils.config.models import ModelConfig',
    r'from sifaka\.utils\.config import OpenAIConfig': 'from sifaka.utils.config.models import OpenAIConfig',
    r'from sifaka\.utils\.config import AnthropicConfig': 'from sifaka.utils.config.models import AnthropicConfig',
    r'from sifaka\.utils\.config import GeminiConfig': 'from sifaka.utils.config.models import GeminiConfig',
    r'from sifaka\.utils\.config import standardize_model_config': 'from sifaka.utils.config.models import standardize_model_config',
    
    # Rule configs
    r'from sifaka\.utils\.config import RuleConfig': 'from sifaka.utils.config.rules import RuleConfig',
    r'from sifaka\.utils\.config import RulePriority': 'from sifaka.utils.config.rules import RulePriority',
    r'from sifaka\.utils\.config import standardize_rule_config': 'from sifaka.utils.config.rules import standardize_rule_config',
    
    # Critic configs
    r'from sifaka\.utils\.config import CriticConfig': 'from sifaka.utils.config.critics import CriticConfig',
    r'from sifaka\.utils\.config import CriticMetadata': 'from sifaka.utils.config.critics import CriticMetadata',
    r'from sifaka\.utils\.config import PromptCriticConfig': 'from sifaka.utils.config.critics import PromptCriticConfig',
    r'from sifaka\.utils\.config import ReflexionCriticConfig': 'from sifaka.utils.config.critics import ReflexionCriticConfig',
    r'from sifaka\.utils\.config import ConstitutionalCriticConfig': 'from sifaka.utils.config.critics import ConstitutionalCriticConfig',
    r'from sifaka\.utils\.config import SelfRefineCriticConfig': 'from sifaka.utils.config.critics import SelfRefineCriticConfig',
    r'from sifaka\.utils\.config import SelfRAGCriticConfig': 'from sifaka.utils.config.critics import SelfRAGCriticConfig',
    r'from sifaka\.utils\.config import FeedbackCriticConfig': 'from sifaka.utils.config.critics import FeedbackCriticConfig',
    r'from sifaka\.utils\.config import ValueCriticConfig': 'from sifaka.utils.config.critics import ValueCriticConfig',
    r'from sifaka\.utils\.config import LACCriticConfig': 'from sifaka.utils.config.critics import LACCriticConfig',
    r'from sifaka\.utils\.config import standardize_critic_config': 'from sifaka.utils.config.critics import standardize_critic_config',
    
    # Chain configs
    r'from sifaka\.utils\.config import ChainConfig': 'from sifaka.utils.config.chain import ChainConfig',
    r'from sifaka\.utils\.config import EngineConfig': 'from sifaka.utils.config.chain import EngineConfig',
    r'from sifaka\.utils\.config import ValidatorConfig': 'from sifaka.utils.config.chain import ValidatorConfig',
    r'from sifaka\.utils\.config import ImproverConfig': 'from sifaka.utils.config.chain import ImproverConfig',
    r'from sifaka\.utils\.config import standardize_chain_config': 'from sifaka.utils.config.chain import standardize_chain_config',
    
    # Classifier configs
    r'from sifaka\.utils\.config import ClassifierConfig': 'from sifaka.utils.config.classifiers import ClassifierConfig',
    r'from sifaka\.utils\.config import extract_classifier_config_params': 'from sifaka.utils.config.classifiers import extract_classifier_config_params',
    r'from sifaka\.utils\.config import standardize_classifier_config': 'from sifaka.utils.config.classifiers import standardize_classifier_config',
    
    # Retrieval configs
    r'from sifaka\.utils\.config import RetrieverConfig': 'from sifaka.utils.config.retrieval import RetrieverConfig',
    r'from sifaka\.utils\.config import QueryProcessingConfig': 'from sifaka.utils.config.retrieval import QueryProcessingConfig',
    r'from sifaka\.utils\.config import RankingConfig': 'from sifaka.utils.config.retrieval import RankingConfig',
    r'from sifaka\.utils\.config import IndexConfig': 'from sifaka.utils.config.retrieval import IndexConfig',
    r'from sifaka\.utils\.config import standardize_retriever_config': 'from sifaka.utils.config.retrieval import standardize_retriever_config',
}

ERROR_MAPPINGS = {
    # Base errors
    r'from sifaka\.utils\.errors import SifakaError': 'from sifaka.utils.errors.base import SifakaError',
    r'from sifaka\.utils\.errors import ValidationError': 'from sifaka.utils.errors.base import ValidationError',
    r'from sifaka\.utils\.errors import ConfigurationError': 'from sifaka.utils.errors.base import ConfigurationError',
    r'from sifaka\.utils\.errors import ProcessingError': 'from sifaka.utils.errors.base import ProcessingError',
    r'from sifaka\.utils\.errors import ResourceError': 'from sifaka.utils.errors.base import ResourceError',
    r'from sifaka\.utils\.errors import TimeoutError': 'from sifaka.utils.errors.base import TimeoutError',
    r'from sifaka\.utils\.errors import InputError': 'from sifaka.utils.errors.base import InputError',
    r'from sifaka\.utils\.errors import StateError': 'from sifaka.utils.errors.base import StateError',
    r'from sifaka\.utils\.errors import DependencyError': 'from sifaka.utils.errors.base import DependencyError',
    r'from sifaka\.utils\.errors import InitializationError': 'from sifaka.utils.errors.base import InitializationError',
    r'from sifaka\.utils\.errors import CleanupError': 'from sifaka.utils.errors.base import CleanupError',
    r'from sifaka\.utils\.errors import ComponentError': 'from sifaka.utils.errors.base import ComponentError',
    
    # Component errors
    r'from sifaka\.utils\.errors import ChainError': 'from sifaka.utils.errors.component import ChainError',
    r'from sifaka\.utils\.errors import ModelError': 'from sifaka.utils.errors.component import ModelError',
    r'from sifaka\.utils\.errors import RuleError': 'from sifaka.utils.errors.component import RuleError',
    r'from sifaka\.utils\.errors import CriticError': 'from sifaka.utils.errors.component import CriticError',
    r'from sifaka\.utils\.errors import ClassifierError': 'from sifaka.utils.errors.component import ClassifierError',
    r'from sifaka\.utils\.errors import RetrievalError': 'from sifaka.utils.errors.component import RetrievalError',
    r'from sifaka\.utils\.errors import ImproverError': 'from sifaka.utils.errors.component import ImproverError',
    r'from sifaka\.utils\.errors import FormatterError': 'from sifaka.utils.errors.component import FormatterError',
    r'from sifaka\.utils\.errors import PluginError': 'from sifaka.utils.errors.component import PluginError',
    r'from sifaka\.utils\.errors import ImplementationError': 'from sifaka.utils.errors.component import ImplementationError',
    
    # Error handling
    r'from sifaka\.utils\.errors import handle_error': 'from sifaka.utils.errors.handling import handle_error',
    r'from sifaka\.utils\.errors import try_operation': 'from sifaka.utils.errors.handling import try_operation',
    r'from sifaka\.utils\.errors import try_component_operation': 'from sifaka.utils.errors.handling import try_component_operation',
    r'from sifaka\.utils\.errors import create_error_handler': 'from sifaka.utils.errors.handling import create_error_handler',
    
    # Error results
    r'from sifaka\.utils\.errors import ErrorResult': 'from sifaka.utils.errors.results import ErrorResult',
    r'from sifaka\.utils\.errors import create_error_result': 'from sifaka.utils.errors.results import create_error_result',
    r'from sifaka\.utils\.errors import create_error_result_factory': 'from sifaka.utils.errors.results import create_error_result_factory',
    r'from sifaka\.utils\.errors import create_chain_error_result': 'from sifaka.utils.errors.results import create_chain_error_result',
    r'from sifaka\.utils\.errors import create_model_error_result': 'from sifaka.utils.errors.results import create_model_error_result',
    r'from sifaka\.utils\.errors import create_rule_error_result': 'from sifaka.utils.errors.results import create_rule_error_result',
    r'from sifaka\.utils\.errors import create_critic_error_result': 'from sifaka.utils.errors.results import create_critic_error_result',
    r'from sifaka\.utils\.errors import create_classifier_error_result': 'from sifaka.utils.errors.results import create_classifier_error_result',
    
    # Safe execution
    r'from sifaka\.utils\.errors import safely_execute_component_operation': 'from sifaka.utils.errors.safe_execution import safely_execute_component_operation',
    r'from sifaka\.utils\.errors import safely_execute_rule': 'from sifaka.utils.errors.safe_execution import safely_execute_rule',
    r'from sifaka\.utils\.errors import safely_execute_chain': 'from sifaka.utils.errors.safe_execution import safely_execute_chain',
    r'from sifaka\.utils\.errors import safely_execute_component': 'from sifaka.utils.errors.safe_execution import safely_execute_component',
    
    # Error logging
    r'from sifaka\.utils\.errors import log_error': 'from sifaka.utils.errors.logging import log_error',
}

DEPENDENCY_MAPPINGS = {
    # Provider
    r'from sifaka\.core\.dependency import DependencyProvider': 'from sifaka.core.dependency.provider import DependencyProvider',
    
    # Scopes
    r'from sifaka\.core\.dependency import DependencyScope': 'from sifaka.core.dependency.scopes import DependencyScope',
    r'from sifaka\.core\.dependency import SessionScope': 'from sifaka.core.dependency.scopes import SessionScope',
    r'from sifaka\.core\.dependency import RequestScope': 'from sifaka.core.dependency.scopes import RequestScope',
    
    # Injector
    r'from sifaka\.core\.dependency import DependencyInjector': 'from sifaka.core.dependency.injector import DependencyInjector',
    r'from sifaka\.core\.dependency import inject_dependencies': 'from sifaka.core.dependency.injector import inject_dependencies',
    
    # Utils
    r'from sifaka\.core\.dependency import provide_dependency': 'from sifaka.core.dependency.utils import provide_dependency',
    r'from sifaka\.core\.dependency import provide_factory': 'from sifaka.core.dependency.utils import provide_factory',
    r'from sifaka\.core\.dependency import get_dependency': 'from sifaka.core.dependency.utils import get_dependency',
    r'from sifaka\.core\.dependency import get_dependency_by_type': 'from sifaka.core.dependency.utils import get_dependency_by_type',
    r'from sifaka\.core\.dependency import create_session_scope': 'from sifaka.core.dependency.utils import create_session_scope',
    r'from sifaka\.core\.dependency import create_request_scope': 'from sifaka.core.dependency.utils import create_request_scope',
    r'from sifaka\.core\.dependency import clear_dependencies': 'from sifaka.core.dependency.utils import clear_dependencies',
}

# Combine all mappings
ALL_MAPPINGS = {**CONFIG_MAPPINGS, **ERROR_MAPPINGS, **DEPENDENCY_MAPPINGS}

def update_imports(file_path: Path, dry_run: bool = False, verbose: bool = False) -> Tuple[bool, List[str]]:
    """Update imports in a single file.
    
    Args:
        file_path: Path to the file to update
        dry_run: If True, don't write changes
        verbose: If True, print detailed information
        
    Returns:
        Tuple of (was_updated, list_of_changes)
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False, []
    
    original_content = content
    changes = []
    
    # Apply each mapping
    for pattern, replacement in ALL_MAPPINGS.items():
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            changes.append(f"{pattern} -> {replacement}")
    
    # Handle multi-import statements
    # This is more complex and would require parsing the Python code
    # For now, we'll just note these as potential manual fixes
    
    if content != original_content:
        if not dry_run:
            try:
                with open(file_path, 'w') as f:
                    f.write(content)
            except Exception as e:
                print(f"Error writing {file_path}: {e}")
                return False, changes
        
        if verbose:
            print(f"Updated {file_path}")
            for change in changes:
                print(f"  - {change}")
        
        return True, changes
    
    return False, []

def process_directory(directory: Path, dry_run: bool = False, verbose: bool = False) -> Tuple[int, Dict[str, List[str]]]:
    """Process all Python files in a directory recursively.
    
    Args:
        directory: Directory to process
        dry_run: If True, don't write changes
        verbose: If True, print detailed information
        
    Returns:
        Tuple of (number_of_updated_files, dict_of_files_and_changes)
    """
    updated_files = {}
    count = 0
    
    for py_file in directory.glob('**/*.py'):
        updated, changes = update_imports(py_file, dry_run, verbose)
        if updated:
            count += 1
            updated_files[str(py_file)] = changes
    
    return count, updated_files

def main():
    """Main function."""
    # Parse arguments
    dry_run = '--dry-run' in sys.argv
    verbose = '--verbose' in sys.argv
    
    # Get path
    path = '.'
    for arg in sys.argv[1:]:
        if not arg.startswith('--'):
            path = arg
            break
    
    # Process directory
    directory = Path(path)
    if not directory.exists():
        print(f"Error: {directory} does not exist")
        sys.exit(1)
    
    print(f"Processing {directory}...")
    if dry_run:
        print("Dry run mode: no changes will be written")
    
    count, updated_files = process_directory(directory, dry_run, verbose)
    
    print(f"Updated {count} files")
    if not verbose and count > 0:
        print("Files updated:")
        for file_path in updated_files:
            print(f"  - {file_path}")
    
    if count > 0 and not dry_run:
        print("\nRemember to run tests to ensure nothing broke!")
    
    # Report potential manual fixes needed
    print("\nPotential manual fixes needed:")
    print("1. Multi-import statements (e.g., from sifaka.utils.config import (BaseConfig, ModelConfig))")
    print("2. Imports inside functions or conditional imports")
    print("3. Import statements with aliases (e.g., from sifaka.utils.config.base import BaseConfig as BC)")
    print("4. Dynamic imports (e.g., using importlib)")

if __name__ == "__main__":
    main()
