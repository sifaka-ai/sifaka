#!/usr/bin/env python3
"""
Script to fix specific syntax errors in the Sifaka codebase.

This script targets specific files with known syntax errors:
1. sifaka/chain/engine.py - Mismatched parentheses
2. sifaka/chain/chain.py - Various state manager access patterns
3. sifaka/adapters/pydantic_ai/factory.py - Unclosed parentheses
4. sifaka/adapters/guardrails/adapter.py - State manager access patterns
"""

import os
import re
import sys
from typing import Dict, List, Tuple

# Specific file fixes
FILE_FIXES = {
    "sifaka/classifiers/implementations/content/bias.py": [
        # Fix for import statements
        (
            r"from sifaka\.utils\.config and config and config and config\.classifiers import ClassifierConfig",
            r"from sifaka.utils.config.classifiers import ClassifierConfig",
        ),
        # Fix for import statements
        (
            r"from sifaka\.utils\.config and config and config and config\.classifiers import extract_classifier_config_params",
            r"from sifaka.utils.config.classifiers import extract_classifier_config_params",
        ),
        # Fix for ClassifierConfig initialization
        (
            r"config = ClassifierConfig\(\s*labels=\(params and params\.get\(\"bias_types\", DEFAULT_BIAS_TYPES\),\s*cost=self\.DEFAULT_COST,\s*min_confidence=\(params and params\.get\(\"min_confidence\", 0\.7\),\s*params=params,\s*\)",
            r"config = ClassifierConfig(\n                labels=params.get(\"bias_types\", DEFAULT_BIAS_TYPES),\n                cost=self.DEFAULT_COST,\n                min_confidence=params.get(\"min_confidence\", 0.7),\n                params=params,\n            )",
        ),
        # Fix for importlib modules
        (
            r"\"feature_extraction_text\": \(importlib and \(importlib and importlib\.import_module\(\s*\"sklearn\.feature_extraction\.text\"\s*\),",
            r"\"feature_extraction_text\": importlib.import_module(\"sklearn.feature_extraction.text\"),",
        ),
        # Fix for importlib modules
        (
            r"\"svm\": \(importlib and \(importlib and importlib\.import_module\(\"sklearn\.svm\"\),",
            r"\"svm\": importlib.import_module(\"sklearn.svm\"),",
        ),
        # Fix for importlib modules
        (
            r"\"pipeline\": \(importlib and \(importlib and importlib\.import_module\(\"sklearn\.pipeline\"\),",
            r"\"pipeline\": importlib.import_module(\"sklearn.pipeline\"),",
        ),
        # Fix for importlib modules
        (
            r"\"calibration\": \(importlib and \(importlib and importlib\.import_module\(\"sklearn\.calibration\"\),",
            r"\"calibration\": importlib.import_module(\"sklearn.calibration\"),",
        ),
    ],
    "sifaka/chain/state.py": [
        # Fix for len(self.get())
        (
            r"\"cache_size\": lenself\.get\(\"result_cache\", {}\)\),",
            r"\"cache_size\": len(self.get(\"result_cache\", {})),",
        ),
    ],
    "sifaka/chain/factories.py": [
        # Fix for provider.get
        (
            r"model = \(provider and provider\.get\('model_provider', None, session_id,\s*request_id\)",
            r"model = provider.get('model_provider', None, session_id, request_id)",
        ),
        # Fix for provider.get_by_type
        (
            r"model = \(provider and provider\.get_by_type\(ModelProvider, None,\s*session_id, request_id\)",
            r"model = provider.get_by_type(ModelProvider, None, session_id, request_id)",
        ),
        # Fix for critic provider.get
        (
            r"critic = \(provider and provider\.get\('critic', None, session_id,\s*request_id\)",
            r"critic = provider.get('critic', None, session_id, request_id)",
        ),
        # Fix for kwargs.get
        (
            r"name=\(kwargs and kwargs\.get\(\s*'name', '[^']+'\), description=\(kwargs and kwargs\.get\('description',\s*'[^']+'\)",
            r"name=kwargs.get('name', 'simple_chain'), description=kwargs.get('description', 'Simple chain for text generation and validation')",
        ),
        # Fix for backoff chain kwargs.get
        (
            r"name=\(kwargs and kwargs\.get\(\s*'name', 'backoff_chain'\), description=\(kwargs and kwargs\.get\('description',\s*'Chain with backoff retry strategy'\)",
            r"name=kwargs.get('name', 'backoff_chain'), description=kwargs.get('description', 'Chain with backoff retry strategy')",
        ),
    ],
    "sifaka/chain/engine.py": [
        # Fix for component_type metadata
        (
            r"self\.\(_state_manager and _state_manager\.set_metadata\('component_type', self\.__class__\.__name__\)",
            r"self._state_manager.set_metadata('component_type', self.__class__.__name__)",
        ),
        # Fix for execution_start_time metadata
        (
            r"self\.\(_state_manager and _state_manager\.set_metadata\('execution_start_time', start_time\s*\)",
            r"self._state_manager.set_metadata('execution_start_time', start_time)",
        ),
        # Fix for retry_manager execute_with_retries
        (
            r"result = self\.\(_retry_manager and _retry_manager\.execute_with_retries\(generate_func\s*=lambda\s*:\s*self\._generate_output\(prompt\),\s*validate_func=\s*lambda output:\s*self\._validate_output\(output\),\s*improve_func=lambda output,\s*results:\s*self\.\s*_improve_output\(output,\s*results\),\s*prompt=prompt,\s*create_result_func=self\._create_result\)",
            r"result = self._retry_manager.execute_with_retries(\n                    generate_func=lambda: self._generate_output(prompt), \n                    validate_func=lambda output: self._validate_output(output),\n                    improve_func=lambda output, results: self._improve_output(output, results), \n                    prompt=prompt,\n                    create_result_func=self._create_result)",
        ),
        # Fix for last_execution_time and related metadata
        (
            r"self\.\(_state_manager and _state_manager\.set_metadata\('last_execution_time',\s*execution_time\)\s*avg_time = self\.\(_state_manager and _state_manager\.get_metadata\(\s*'avg_execution_time', 0\)\s*count = execution_count \+ 1\s*new_avg = \(avg_time \* \(count - 1\) \+ execution_time\) / count\s*self\._state_manager\.set_metadata\('avg_execution_time', new_avg\)\s*max_time = self\.\(_state_manager and _state_manager\.get_metadata\(\s*'max_execution_time', 0\)\s*if execution_time > max_time:\s*self\.\(_state_manager and _state_manager\.set_metadata\('max_execution_time',\s*execution_time\)",
            r"self._state_manager.set_metadata('last_execution_time', execution_time)\n                avg_time = self._state_manager.get_metadata('avg_execution_time', 0)\n                count = execution_count + 1\n                new_avg = (avg_time * (count - 1) + execution_time) / count\n                self._state_manager.set_metadata('avg_execution_time', new_avg)\n                max_time = self._state_manager.get_metadata('max_execution_time', 0)\n                if execution_time > max_time:\n                    self._state_manager.set_metadata('max_execution_time', execution_time)",
        ),
        # Fix for last_error_time
        (
            r"self\._state_manager\.set_metadata\('last_error_time', time\.time\(\)",
            r"self._state_manager.set_metadata('last_error_time', time.time())",
        ),
        # Fix for execution_start_time in _create_result
        (
            r"start_time = self\.\(_state_manager\.get_metadata\('execution_start_time', 0\s*\)",
            r"start_time = self._state_manager.get_metadata('execution_start_time', 0)",
        ),
        # Fix for config.params
        (
            r"self\.config\.\(params and params\.get\('fail_fast', False\s*\) and not result\.passed",
            r"self.config.params.get('fail_fast', False) and not result.passed",
        ),
        # Fix for metadata in _create_result
        (
            r"'Chain execution completed with validation failures', metadata=\s*{'engine_config': {'max_attempts': self\.config\.max_attempts,\s*'params': self\.config\.params}, 'execution_count': self\.\s*\(_state_manager\.get\('execution_count'\)}",
            r"'Chain execution completed with validation failures', metadata={\n                'engine_config': {\n                    'max_attempts': self.config.max_attempts,\n                    'params': self.config.params\n                }, \n                'execution_count': self._state_manager.get('execution_count')\n            }",
        ),
        # Fix for formatter.format
        (
            r"return \(formatter\.format\(output, validation_results\)",
            r"return formatter.format(output, validation_results)",
        ),
        # Fix for logger.warning
        (
            r"\(logger\.warning\(f'Result formatting failed: {str\(e\)}'\)",
            r"logger.warning(f'Result formatting failed: {str(e)}')",
        ),
    ],
    "sifaka/chain/chain.py": [
        # Fix for get_statistics method
        (
            r"return {'name': self\._name, 'execution_count': self\._state_manager\.\s*get\('execution_count', 0\), 'success_count': self\._state_manager\s*\.get_metadata\('success_count', 0\), 'failure_count': self\.\s*\(_state_manager and _state_manager\.get_metadata\('failure_count', 0\), 'error_count':\s*self\._state_manager\.get_metadata\('error_count', 0\),\s*'avg_execution_time': self\.\(_state_manager and _state_manager\.get_metadata\(\s*'avg_execution_time', 0\), 'max_execution_time': self\.\s*\(_state_manager and _state_manager\.get_metadata\('max_execution_time', 0\),\s*'last_execution_time': self\.\(_state_manager and _state_manager\.get_metadata\(\s*'last_execution_time', 0\), 'last_error': self\._state_manager\.\s*get_metadata\('last_error', None\), 'last_error_time': self\.\s*\(_state_manager and _state_manager\.get_metadata\('last_error_time', None\),\s*'cache_size': len\(self\._state_manager\.get\('result_cache', {}\)\)}",
            r"return {\n            'name': self._name, \n            'execution_count': self._state_manager.get('execution_count', 0), \n            'success_count': self._state_manager.get_metadata('success_count', 0), \n            'failure_count': self._state_manager.get_metadata('failure_count', 0), \n            'error_count': self._state_manager.get_metadata('error_count', 0),\n            'avg_execution_time': self._state_manager.get_metadata('avg_execution_time', 0), \n            'max_execution_time': self._state_manager.get_metadata('max_execution_time', 0),\n            'last_execution_time': self._state_manager.get_metadata('last_execution_time', 0), \n            'last_error': self._state_manager.get_metadata('last_error', None), \n            'last_error_time': self._state_manager.get_metadata('last_error_time', None),\n            'cache_size': len(self._state_manager.get('result_cache', {}))\n        }",
        ),
    ],
    "sifaka/adapters/pydantic_ai/factory.py": [
        # Fix for config_dict
        (r"config_dict = \(config\.copy\(\)", r"config_dict = config.copy()"),
        # Fix for adapter._state_manager
        (
            r"adapter\.\(_state_manager and _state_manager\.set_metadata\('name', name\)",
            r"adapter._state_manager.set_metadata('name', name)",
        ),
        # Fix for adapter.warm_up
        (r"\(adapter\.warm_up\(\)", r"adapter.warm_up()"),
        # Fix for logger.debug
        (r"\(logger\.debug\(", r"logger.debug("),
    ],
    "sifaka/adapters/guardrails/adapter.py": [
        # Fix for get_detailed_statistics
        (
            r"\(stats and stats\.update\({'validator_type': self\.\(_state_manager and _state_manager\.get_metadata\(\s*'validator_type', 'unknown'\), 'validator_class': self\.adaptee\.\s*__class__\.__name__}\)",
            r"if stats:\n            stats.update({'validator_type': self._state_manager.get_metadata(\n                'validator_type', 'unknown'), 'validator_class': self.adaptee.__class__.__name__})",
        ),
        # Fix for state manager metadata
        (
            r"self\.\(_state_manager and _state_manager\.set_metadata\('([^']+)', ([^)]+)\)",
            r"self._state_manager.set_metadata('\1', \2)",
        ),
        # Fix for state.adaptee
        (r"state\.\(adaptee and adaptee\.validate\(([^)]+)\)", r"state.adaptee.validate(\1)"),
    ],
}


def fix_specific_file(file_path: str) -> Tuple[int, List[str]]:
    """
    Apply specific fixes to a file.

    Args:
        file_path: Path to the file to fix

    Returns:
        Tuple of (number of fixes, list of fixed patterns)
    """
    if file_path not in FILE_FIXES:
        return 0, []

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    original_content = content
    fixes_applied = []

    for pattern, replacement in FILE_FIXES[file_path]:
        # Apply the fix
        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        if new_content != content:
            content = new_content
            fixes_applied.append(f"Applied specific fix")

    # Only write back if changes were made
    if content != original_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    return len(fixes_applied), fixes_applied


def main():
    """Main function to run the script."""
    total_fixes = 0
    files_fixed = 0

    for file_path in FILE_FIXES.keys():
        if os.path.exists(file_path):
            num_fixes, fixes = fix_specific_file(file_path)
            if num_fixes > 0:
                files_fixed += 1
                total_fixes += num_fixes
                print(f"Fixed {num_fixes} issues in {file_path}")
                for fix in fixes:
                    print(f"  - {fix}")
        else:
            print(f"Warning: File {file_path} does not exist")

    print(f"\nSummary: Fixed {total_fixes} issues in {files_fixed} files")


if __name__ == "__main__":
    main()
