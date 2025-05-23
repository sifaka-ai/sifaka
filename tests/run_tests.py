#!/usr/bin/env python3
"""
Test runner for Sifaka persistence tests.

This script runs all persistence-related tests and provides a summary report.
"""

import sys
import time
import importlib.util
from pathlib import Path

# Project root for reference
project_root = Path(__file__).parent.parent


def run_test_module(module_path: Path) -> tuple[bool, float, str]:
    """
    Run a test module and return results.

    Args:
        module_path: Path to the test module

    Returns:
        Tuple of (success, duration, output)
    """
    print(f"\n{'='*60}")
    print(f"Running {module_path.name}")
    print("=" * 60)

    start_time = time.time()

    try:
        # Load and run the module
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec is None:
            raise ImportError(f"Could not load spec for {module_path.stem} from {module_path}")

        module = importlib.util.module_from_spec(spec)

        # Capture output
        import io
        import contextlib

        output_buffer = io.StringIO()

        with contextlib.redirect_stdout(output_buffer):
            if spec.loader is None:
                raise ImportError(f"No loader found for {module_path.stem}")
            spec.loader.exec_module(module)
            if hasattr(module, "main"):
                result = module.main()
            else:
                result = 0

        duration = time.time() - start_time
        output = output_buffer.getvalue()

        # Print the captured output
        print(output)

        success = result == 0
        return success, duration, output

    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"❌ Error running {module_path.name}: {e}"
        print(error_msg)
        import traceback

        traceback.print_exc()
        return False, duration, error_msg


def main() -> int:
    """Run all tests and provide summary."""
    print("🧪 Sifaka Persistence Test Suite")
    print("=" * 60)

    # Find all test files
    test_dir = Path(__file__).parent
    test_files = [
        test_dir / "test_smoke.py",
        test_dir / "test_persistence.py",
        test_dir / "test_thought_history.py",
    ]

    # Filter to only existing files
    test_files = [f for f in test_files if f.exists()]

    if not test_files:
        print("❌ No test files found!")
        return 1

    print(f"Found {len(test_files)} test files:")
    for test_file in test_files:
        print(f"  - {test_file.name}")

    # Run all tests
    results = []
    total_start_time = time.time()

    for test_file in test_files:
        success, duration, output = run_test_module(test_file)
        results.append((test_file.name, success, duration, output))

    total_duration = time.time() - total_start_time

    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success, _, _ in results if success)
    failed = len(results) - passed

    print(f"Total tests run: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_duration:.2f}s")

    print("\nDetailed Results:")
    for name, success, duration, _ in results:
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"  {status} {name:<25} ({duration:.2f}s)")

    if failed > 0:
        print(f"\n❌ {failed} test(s) failed!")
        return 1
    else:
        print(f"\n🎉 All {passed} tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
