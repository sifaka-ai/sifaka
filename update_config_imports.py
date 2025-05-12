#!/usr/bin/env python3
"""
Script to update imports from sifaka.utils.config to specific modules.

This script updates imports from the old module structure to the new modular directory structure:
- sifaka.utils.config -> sifaka.utils.config.{base,models,rules,critics,chain,classifiers,retrieval}

Usage:
    python update_config_imports.py
"""

import os
import re
from pathlib import Path

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
    r'from sifaka\.utils\.config import standardize_classifier_config': 'from sifaka.utils.config.classifiers import standardize_classifier_config',
    r'from sifaka\.utils\.config import extract_classifier_config_params': 'from sifaka.utils.config.classifiers import extract_classifier_config_params',
    
    # Retrieval configs
    r'from sifaka\.utils\.config import RetrieverConfig': 'from sifaka.utils.config.retrieval import RetrieverConfig',
    r'from sifaka\.utils\.config import QueryProcessingConfig': 'from sifaka.utils.config.retrieval import QueryProcessingConfig',
    r'from sifaka\.utils\.config import RankingConfig': 'from sifaka.utils.config.retrieval import RankingConfig',
    r'from sifaka\.utils\.config import IndexConfig': 'from sifaka.utils.config.retrieval import IndexConfig',
    r'from sifaka\.utils\.config import standardize_retriever_config': 'from sifaka.utils.config.retrieval import standardize_retriever_config',
    
    # Default configs
    r'from sifaka\.utils\.config import DEFAULT_PROMPT_CONFIG': 'from sifaka.utils.config.critics import DEFAULT_PROMPT_CONFIG',
    r'from sifaka\.utils\.config import DEFAULT_REFLEXION_CONFIG': 'from sifaka.utils.config.critics import DEFAULT_REFLEXION_CONFIG',
    r'from sifaka\.utils\.config import DEFAULT_CONSTITUTIONAL_CONFIG': 'from sifaka.utils.config.critics import DEFAULT_CONSTITUTIONAL_CONFIG',
    r'from sifaka\.utils\.config import DEFAULT_SELF_REFINE_CONFIG': 'from sifaka.utils.config.critics import DEFAULT_SELF_REFINE_CONFIG',
    r'from sifaka\.utils\.config import DEFAULT_SELF_RAG_CONFIG': 'from sifaka.utils.config.critics import DEFAULT_SELF_RAG_CONFIG',
    r'from sifaka\.utils\.config import DEFAULT_FEEDBACK_CONFIG': 'from sifaka.utils.config.critics import DEFAULT_FEEDBACK_CONFIG',
    r'from sifaka\.utils\.config import DEFAULT_VALUE_CONFIG': 'from sifaka.utils.config.critics import DEFAULT_VALUE_CONFIG',
    r'from sifaka\.utils\.config import DEFAULT_LAC_CONFIG': 'from sifaka.utils.config.critics import DEFAULT_LAC_CONFIG',
}

def update_imports(file_path):
    """Update imports in a file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Apply each mapping
    for old_import, new_import in CONFIG_MAPPINGS.items():
        content = re.sub(old_import, new_import, content)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)

def main():
    """Main function."""
    # Find all Python files in the sifaka directory
    sifaka_dir = Path('sifaka')
    tests_dir = Path('tests')
    
    # Process all Python files in sifaka and tests directories
    for directory in [sifaka_dir, tests_dir]:
        for file_path in directory.glob('**/*.py'):
            update_imports(file_path)
            print(f"Updated imports in {file_path}")

if __name__ == '__main__':
    main()
