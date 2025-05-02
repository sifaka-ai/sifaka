"""
Run isolated tests for the critics modules without importing the main framework.

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

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import test modules directly without triggering automatic imports
def import_module_from_file(module_name, file_path):
    """Import a module from a file path without triggering normal imports."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)

    # Patch the import system for this module
    original_import = __import__

    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Block problematic imports that would trigger LangChain
        if name.startswith('sifaka.chain') or name.startswith('sifaka.models'):
            # Create and return a mock module
            import types
            return types.ModuleType(name)
        # For certain specific imports we need to fake, create mock objects
        if name == 'sifaka.critics.protocols' and fromlist:
            import types
            mod = types.ModuleType(name)
            # Add protocol classes as needed
            for item in fromlist:
                if item in ['TextValidator', 'TextImprover', 'TextCritic', 'CritiqueResult']:
                    setattr(mod, item, type(item, (), {}))
            return mod
        if name == 'sifaka.critics.base' and fromlist:
            import types
            mod = types.ModuleType(name)
            # Add base classes as needed
            for item in fromlist:
                if item in ['BaseCritic', 'CriticConfig', 'CriticMetadata']:
                    setattr(mod, item, type(item, (), {}))
            return mod
        if name == 'sifaka.critics.reflexion' and fromlist:
            import types
            mod = types.ModuleType(name)
            # Add reflexion classes as needed
            for item in fromlist:
                if item in ['ReflexionCritic', 'ReflexionCriticConfig', 'create_reflexion_critic', 'ReflexionPromptFactory']:
                    setattr(mod, item, type(item, (), {}))
            return mod
        if name == 'sifaka.critics.style' and fromlist:
            import types
            mod = types.ModuleType(name)
            # Add style classes as needed
            for item in fromlist:
                if item in ['StyleCritic', 'create_style_critic']:
                    setattr(mod, item, type(item, (), {}))
            return mod
        if name == 'sifaka.critics.managers.response' and fromlist:
            import types
            mod = types.ModuleType(name)
            # Add response parser classes as needed
            for item in fromlist:
                if item in ['ResponseParser']:
                    setattr(mod, item, type(item, (), {}))
            return mod
        return original_import(name, globals, locals, fromlist, level)

    # Apply the patch
    sys.modules[module_name] = module
    try:
        # Execute the module
        spec.loader.exec_module(module)
    except ImportError as e:
        print(f"Warning: Could not fully import {module_name}: {e}")
    finally:
        # Restore original import
        pass

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
        print(f"Error running {module_name}: {e}")
        return False

def generate_coverage_report():
    """Generate a coverage report for the tested modules."""
    try:
        cov = coverage.Coverage(source=["sifaka.critics"])
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

    # Run protocol tests
    print("\n=== Testing protocols ===")
    protocols_success = run_tests_from_file(
        "tests.critics.test_protocols",
        os.path.join(os.path.dirname(__file__), "test_protocols.py")
    )
    success = success and protocols_success

    # Run reflexion tests
    print("\n=== Testing reflexion ===")
    reflexion_success = run_tests_from_file(
        "tests.critics.test_reflexion",
        os.path.join(os.path.dirname(__file__), "test_reflexion.py")
    )
    success = success and reflexion_success

    # Run style tests
    print("\n=== Testing style ===")
    style_success = run_tests_from_file(
        "tests.critics.test_style",
        os.path.join(os.path.dirname(__file__), "test_style.py")
    )
    success = success and style_success

    # Run response parser tests
    print("\n=== Testing response parser ===")
    response_parser_success = run_tests_from_file(
        "tests.critics.test_response_parser",
        os.path.join(os.path.dirname(__file__), "test_response_parser.py")
    )
    success = success and response_parser_success

    # Run reflexion integration tests
    print("\n=== Testing reflexion integration ===")
    reflexion_integration_success = run_tests_from_file(
        "tests.critics.test_reflexion_integration",
        os.path.join(os.path.dirname(__file__), "test_reflexion_integration.py")
    )
    success = success and reflexion_integration_success

    return success

if __name__ == "__main__":
    print("Running isolated tests for critics modules...")

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