#!/usr/bin/env python3
"""Test runner for Sifaka classifier tests.

This script runs all classifier unit tests and provides detailed reporting
on test coverage and results.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_path: str = "tests/unit_tests/classifiers", verbose: bool = False, coverage: bool = False):
    """Run classifier tests with optional coverage reporting.
    
    Args:
        test_path: Path to test directory
        verbose: Enable verbose output
        coverage: Enable coverage reporting
    """
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    if coverage:
        cmd.extend([
            "--cov=sifaka.classifiers",
            "--cov-report=html:htmlcov/classifiers",
            "--cov-report=term-missing",
            "--cov-fail-under=80"
        ])
    
    cmd.extend([
        test_path,
        "--tb=short",
        "-x"  # Stop on first failure
    ])
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n" + "=" * 60)
        print("✅ All classifier tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 60)
        print(f"❌ Tests failed with exit code {e.returncode}")
        return False


def run_specific_classifier_test(classifier_name: str, verbose: bool = False):
    """Run tests for a specific classifier.
    
    Args:
        classifier_name: Name of the classifier to test (e.g., 'bias', 'emotion')
        verbose: Enable verbose output
    """
    test_file = f"tests/unit_tests/classifiers/test_{classifier_name}.py"
    
    if not Path(test_file).exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend([test_file, "--tb=short"])
    
    print(f"Running tests for {classifier_name} classifier...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {classifier_name} classifier tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {classifier_name} classifier tests failed with exit code {e.returncode}")
        return False


def list_available_classifiers():
    """List all available classifier tests."""
    test_dir = Path("tests/unit_tests/classifiers")
    
    if not test_dir.exists():
        print("❌ Classifier test directory not found")
        return
    
    test_files = list(test_dir.glob("test_*.py"))
    
    if not test_files:
        print("❌ No classifier test files found")
        return
    
    print("Available classifier tests:")
    print("-" * 30)
    
    for test_file in sorted(test_files):
        classifier_name = test_file.stem.replace("test_", "")
        print(f"  • {classifier_name}")
    
    print(f"\nTotal: {len(test_files)} classifier test files")


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Run Sifaka classifier tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/run_classifier_tests.py                    # Run all classifier tests
  python tests/run_classifier_tests.py --verbose          # Run with verbose output
  python tests/run_classifier_tests.py --coverage         # Run with coverage reporting
  python tests/run_classifier_tests.py --classifier bias  # Run only bias classifier tests
  python tests/run_classifier_tests.py --list             # List available classifiers
        """
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose test output"
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Enable coverage reporting"
    )
    
    parser.add_argument(
        "--classifier",
        type=str,
        help="Run tests for a specific classifier (e.g., 'bias', 'emotion')"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available classifier tests"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_classifiers()
        return
    
    if args.classifier:
        success = run_specific_classifier_test(args.classifier, args.verbose)
    else:
        success = run_tests(verbose=args.verbose, coverage=args.coverage)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
