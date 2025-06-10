#!/usr/bin/env python3
"""Comprehensive test runner for Sifaka.

This script runs the comprehensive test suite including:
- Core component tests (engine, thought, interfaces)
- Graph node tests (generation, validation, critique)
- Validator and critic tests
- Integration and end-to-end tests
- Performance and reliability tests

Usage:
    python tests/run_comprehensive_tests.py [options]

Options:
    --unit: Run unit tests only
    --integration: Run integration tests only
    --performance: Run performance tests only
    --coverage: Generate coverage report
    --verbose: Verbose output
    --fast: Skip slow tests
    --parallel: Run tests in parallel
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


def run_command(cmd: List[str], description: str, timeout: Optional[int] = None) -> bool:
    """Run a command and return success status."""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)

    try:
        start_time = time.time()
        result = subprocess.run(cmd, check=False, timeout=timeout, capture_output=False)
        end_time = time.time()

        duration = end_time - start_time
        if result.returncode == 0:
            print(f"✅ {description} completed successfully in {duration:.2f}s")
            return True
        else:
            print(f"❌ {description} failed with exit code {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"💥 {description} failed with error: {e}")
        return False


def run_unit_tests(verbose: bool = False, coverage: bool = False, fast: bool = False) -> bool:
    """Run comprehensive unit tests."""
    cmd = ["python", "-m", "pytest"]

    # Add test paths for comprehensive unit tests
    test_paths = [
        "tests/unit_tests/core/test_engine.py",
        "tests/unit_tests/core/test_thought_comprehensive.py",
        "tests/unit_tests/core/test_interfaces.py",
        "tests/unit_tests/graph/test_nodes_comprehensive.py",
        "tests/unit_tests/validators/test_validators_comprehensive.py",
    ]
    cmd.extend(test_paths)

    # Add options
    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(
            [
                "--cov=sifaka",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov_comprehensive",
                "--cov-fail-under=80",
            ]
        )

    if fast:
        cmd.extend(["-m", "not slow"])

    cmd.extend(["--strict-markers", "--strict-config", "--tb=short"])

    return run_command(cmd, "Comprehensive Unit Tests", timeout=300)


def run_integration_tests(verbose: bool = False, fast: bool = False) -> bool:
    """Run comprehensive integration tests."""
    cmd = ["python", "-m", "pytest"]

    # Add integration test paths
    test_paths = [
        "tests/integration_tests/test_end_to_end_workflow.py",
    ]
    cmd.extend(test_paths)

    if verbose:
        cmd.append("-v")

    if fast:
        cmd.extend(["-m", "not slow"])

    cmd.extend(["--strict-markers", "--strict-config", "--tb=short"])

    return run_command(cmd, "Comprehensive Integration Tests", timeout=600)


def run_performance_tests(verbose: bool = False) -> bool:
    """Run performance and load tests."""
    cmd = ["python", "-m", "pytest"]

    # Add performance test markers
    cmd.extend(["-m", "performance", "tests/unit_tests/", "tests/integration_tests/"])

    if verbose:
        cmd.append("-v")

    cmd.extend(["--strict-markers", "--strict-config", "--tb=short"])

    return run_command(cmd, "Performance Tests", timeout=900)


def run_existing_tests(verbose: bool = False, fast: bool = False) -> bool:
    """Run existing test suite for comparison."""
    cmd = ["python", "tests/run_unit_tests.py"]

    if verbose:
        cmd.append("--verbose")

    if not fast:
        cmd.append("--coverage")

    return run_command(cmd, "Existing Test Suite", timeout=600)


def check_test_environment() -> bool:
    """Check that test environment is properly set up."""
    print("🔍 Checking test environment...")

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False

    # Check required packages
    required_packages = ["pytest", "pytest-asyncio", "pytest-cov"]
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} not available")
            return False

    # Check test files exist
    test_files = [
        "tests/unit_tests/core/test_engine.py",
        "tests/unit_tests/core/test_thought_comprehensive.py",
        "tests/unit_tests/graph/test_nodes_comprehensive.py",
        "tests/unit_tests/validators/test_validators_comprehensive.py",
        "tests/integration_tests/test_end_to_end_workflow.py",
    ]

    for test_file in test_files:
        if not Path(test_file).exists():
            print(f"❌ Test file missing: {test_file}")
            return False
        print(f"✅ {test_file} found")

    print("✅ Test environment ready")
    return True


def generate_test_report(results: dict) -> None:
    """Generate a comprehensive test report."""
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE TEST REPORT")
    print("=" * 80)

    total_tests = len(results)
    passed_tests = sum(1 for success in results.values() if success)
    failed_tests = total_tests - passed_tests

    print(f"Total test suites: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")

    print("\nDetailed Results:")
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {test_name}")

    if failed_tests == 0:
        print("\n🎉 All comprehensive tests passed!")
    else:
        print(f"\n⚠️  {failed_tests} test suite(s) failed")

    print("\nNext Steps:")
    if failed_tests > 0:
        print("- Review failed test output above")
        print("- Fix failing tests before proceeding")
        print("- Re-run with --verbose for more details")
    else:
        print("- All tests passing! ✨")
        print("- Consider running performance tests")
        print("- Review coverage report in htmlcov_comprehensive/")


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive Sifaka tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tests/run_comprehensive_tests.py                    # Run all tests
    python tests/run_comprehensive_tests.py --unit             # Unit tests only
    python tests/run_comprehensive_tests.py --integration      # Integration tests only
    python tests/run_comprehensive_tests.py --coverage --verbose  # With coverage and verbose output
    python tests/run_comprehensive_tests.py --fast             # Skip slow tests
        """,
    )

    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--existing", action="store_true", help="Run existing test suite")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--no-env-check", action="store_true", help="Skip environment check")

    args = parser.parse_args()

    print("🚀 Sifaka Comprehensive Test Runner")
    print("=" * 80)

    # Check environment unless skipped
    if not args.no_env_check and not check_test_environment():
        print("❌ Environment check failed")
        sys.exit(1)

    # Determine which tests to run
    run_all = not any([args.unit, args.integration, args.performance, args.existing])

    results = {}

    if args.unit or run_all:
        results["Unit Tests"] = run_unit_tests(args.verbose, args.coverage, args.fast)

    if args.integration or run_all:
        results["Integration Tests"] = run_integration_tests(args.verbose, args.fast)

    if args.performance or run_all:
        results["Performance Tests"] = run_performance_tests(args.verbose)

    if args.existing:
        results["Existing Tests"] = run_existing_tests(args.verbose, args.fast)

    # Generate report
    if results:
        generate_test_report(results)

        # Exit with appropriate code
        if all(results.values()):
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        print("No tests selected to run")
        sys.exit(1)


if __name__ == "__main__":
    main()
