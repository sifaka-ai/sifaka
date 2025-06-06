#!/usr/bin/env python3
"""Test runner script for Sifaka unit tests.

This script runs all unit tests and generates coverage reports.
It can be used for development and CI/CD pipelines.

Usage:
    python tests/run_unit_tests.py [options]

Options:
    --verbose, -v: Verbose output
    --coverage, -c: Generate coverage report
    --html: Generate HTML coverage report
    --fail-under: Minimum coverage percentage (default: 80)
    --markers: Run tests with specific markers (e.g., "not slow")
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(
    verbose: bool = False,
    coverage: bool = True,
    html_coverage: bool = False,
    fail_under: int = 80,
    markers: str = None,
    test_path: str = "tests/unit_tests"
) -> int:
    """Run unit tests with optional coverage reporting.
    
    Args:
        verbose: Enable verbose output
        coverage: Generate coverage report
        html_coverage: Generate HTML coverage report
        fail_under: Minimum coverage percentage
        markers: Pytest markers to filter tests
        test_path: Path to test directory
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test path
    cmd.append(test_path)
    
    # Add verbose flag
    if verbose:
        cmd.append("-v")
    
    # Add coverage options
    if coverage:
        cmd.extend([
            "--cov=sifaka",
            "--cov-report=term-missing",
            f"--cov-fail-under={fail_under}"
        ])
        
        if html_coverage:
            cmd.append("--cov-report=html")
    
    # Add marker filtering
    if markers:
        cmd.extend(["-m", markers])
    
    # Add other useful options
    cmd.extend([
        "--strict-markers",
        "--strict-config",
        "--tb=short"
    ])
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 60)
    
    # Run the tests
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Run Sifaka unit tests with coverage reporting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tests/run_unit_tests.py                    # Run all tests with coverage
    python tests/run_unit_tests.py -v                 # Verbose output
    python tests/run_unit_tests.py --html             # Generate HTML coverage
    python tests/run_unit_tests.py -m "not slow"      # Skip slow tests
    python tests/run_unit_tests.py --fail-under 90    # Require 90% coverage
        """
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose test output"
    )
    
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        default=True,
        help="Generate coverage report (default: True)"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage reporting"
    )
    
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML coverage report"
    )
    
    parser.add_argument(
        "--fail-under",
        type=int,
        default=80,
        help="Minimum coverage percentage (default: 80)"
    )
    
    parser.add_argument(
        "-m", "--markers",
        type=str,
        help="Pytest markers to filter tests (e.g., 'not slow')"
    )
    
    parser.add_argument(
        "--test-path",
        type=str,
        default="tests/unit_tests",
        help="Path to test directory (default: tests/unit_tests)"
    )
    
    args = parser.parse_args()
    
    # Handle coverage flags
    coverage = args.coverage and not args.no_coverage
    
    # Verify test directory exists
    test_path = Path(args.test_path)
    if not test_path.exists():
        print(f"Error: Test directory '{test_path}' does not exist")
        return 1
    
    print("Sifaka Unit Test Runner")
    print("=" * 60)
    print(f"Test directory: {test_path}")
    print(f"Coverage enabled: {coverage}")
    print(f"HTML coverage: {args.html}")
    print(f"Minimum coverage: {args.fail_under}%")
    if args.markers:
        print(f"Test markers: {args.markers}")
    print()
    
    # Run the tests
    exit_code = run_tests(
        verbose=args.verbose,
        coverage=coverage,
        html_coverage=args.html,
        fail_under=args.fail_under,
        markers=args.markers,
        test_path=str(test_path)
    )
    
    # Print summary
    print("\n" + "=" * 60)
    if exit_code == 0:
        print("✅ All tests passed!")
        if coverage and args.html:
            print("📊 HTML coverage report generated in htmlcov/")
    else:
        print("❌ Some tests failed or coverage is below threshold")
        print(f"Exit code: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
