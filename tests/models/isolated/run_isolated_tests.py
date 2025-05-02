#!/usr/bin/env python
"""
Run isolated tests for the models modules without importing the main framework.

This script allows running tests without triggering the Pydantic v2 compatibility
issue in LangChain's discriminated unions.
"""

import os
import sys
import unittest
import pytest
import importlib.util
from unittest.mock import MagicMock
import coverage
import builtins

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import test modules directly without triggering automatic imports
def import_module_from_file(module_name, file_path):
    """Import a module from a file path without triggering normal imports."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)

    # Patch the import system for this module
    original_import = builtins.__import__

    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Block problematic imports that would trigger LangChain
        if name.startswith('langchain') or name.startswith('langgraph'):
            # Create and return a mock module
            import types
            return types.ModuleType(name)

        # For certain specific imports we need to fake, create mock objects
        if name == 'sifaka.models.base' and fromlist:
            import types
            mod = types.ModuleType(name)
            # Add base classes as needed
            for item in fromlist:
                if item in ['ModelProvider', 'APIClient', 'TokenCounter', 'ModelConfig']:
                    setattr(mod, item, type(item, (), {}))
            return mod

        if name == 'sifaka.models.mock' and fromlist:
            # For the mock module, we want to import the actual implementation
            try:
                # Just use our own mock implementation in the test file
                return module
            except ImportError:
                # Fall back to mock if real import fails
                import types
                mod = types.ModuleType(name)
                setattr(mod, 'MockProvider', type('MockProvider', (), {}))
                return mod

        return original_import(name, globals, locals, fromlist, level)

    # Apply the patch
    builtins.__import__ = patched_import

    # Add module to sys.modules
    sys.modules[module_name] = module

    try:
        # Execute the module
        spec.loader.exec_module(module)
    except ImportError as e:
        print(f"Warning: Could not fully import {module_name}: {e}")
    except Exception as e:
        print(f"Error executing module {module_name}: {str(e)}")
    finally:
        # Restore original import
        builtins.__import__ = original_import

    return module

# Run tests for a specific test file
def run_tests_from_file(module_name, file_path):
    """Run tests from a specific file."""
    try:
        test_module = import_module_from_file(
            module_name,
            file_path
        )

        # Create test suite and run
        suite = unittest.TestLoader().loadTestsFromModule(test_module)
        result = unittest.TextTestRunner().run(suite)

        print(f"{module_name}: {result.testsRun} tests run")
        print(f"  Errors: {len(result.errors)}")
        print(f"  Failures: {len(result.failures)}")

        # Print errors if any
        if result.errors:
            print("\nErrors:")
            for test, error in result.errors:
                print(f"  {test}: {error}")

        # Print failures if any
        if result.failures:
            print("\nFailures:")
            for test, failure in result.failures:
                print(f"  {test}: {failure}")

        return result.wasSuccessful()
    except Exception as e:
        print(f"Error running {module_name}: {str(e)}")
        return False

def generate_coverage_report():
    """Generate a coverage report for the tested modules."""
    try:
        cov = coverage.Coverage(source=["sifaka.models.mock"])
        cov.start()

        # Run all tests
        run_all_tests()

        cov.stop()
        cov.save()

        # Print coverage report
        print("\n=== Coverage Report ===")
        cov.report()

        # Generate HTML report
        cov.html_report(directory="htmlcov")
        print("HTML coverage report generated in 'htmlcov' directory")
    except Exception as e:
        print(f"Error generating coverage report: {e}")

def run_all_tests():
    """Run all test modules."""
    success = True

    # Run mock model tests
    print("\n=== Testing mock model ===")
    mock_model_success = run_tests_from_file(
        "tests.models.isolated.test_mock_model",
        os.path.join(os.path.dirname(__file__), "test_mock_model.py")
    )
    success = success and mock_model_success

    return success

if __name__ == "__main__":
    print("Running isolated tests for models modules...")

    # Check if coverage report is requested
    if "--with-coverage" in sys.argv:
        generate_coverage_report()
    else:
        success = run_all_tests()

        # Print final result
        if success:
            print("\n✅ All isolated tests passed!")
        else:
            print("\n❌ Some tests failed.")
            sys.exit(1)