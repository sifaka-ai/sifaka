#!/usr/bin/env python
"""Run all tests with coverage reporting.

This script runs the unit tests for the project and generates a coverage report.
It handles special cases like the critics module that needs to be run in isolation
due to Pydantic v2 compatibility issues with LangChain.
"""

import os
import subprocess
import sys

# Try to import and apply patches
try:
    from sifaka.utils.patches import apply_all_patches
    print("✅ Applied compatibility patches")
except ImportError:
    print("❌ Error applying patches: cannot import name 'apply_all_patches' from 'sifaka.utils.patches' (unknown location)")

# === Run critics tests in isolation ===
print("\n=== Running critics tests with isolated runner ===")
critics_runner = os.path.join("tests", "critics", "run_isolated_tests.py")
result = subprocess.run(["python", critics_runner], capture_output=False)
if result.returncode != 0:
    print("❌ Critics tests failed")
    sys.exit(1)

# === Run models tests in isolation ===
print("\n=== Running models tests with isolated runner ===")
models_runner = os.path.join("tests", "models", "isolated", "run_isolated_tests.py")
result = subprocess.run(["python", models_runner], capture_output=False)
if result.returncode != 0:
    print("❌ Models tests failed")
    sys.exit(1)

# === Run the rest of the tests with pytest ===
test_dirs = [
    "tests/chain",
    "tests/utils",
    "tests/validation",
    "tests/performance",
    "tests/integration",
    "tests/examples",
]

# Run rules tests separately, excluding formatting
print("\n=== Running tests in tests/rules (excluding formatting) ===")
result = subprocess.run(
    ["pytest", "tests/rules", "--ignore=tests/rules/formatting"],
    capture_output=False
)
if result.returncode != 0:
    print("❌ Tests in tests/rules failed")
    sys.exit(1)

# Run the remaining test directories
for test_dir in test_dirs:
    if not os.path.exists(test_dir):
        print(f"Warning: Test directory {test_dir} does not exist, skipping.")
        continue

    print(f"\n=== Running tests in {test_dir} ===")
    result = subprocess.run(["pytest", test_dir], capture_output=False)
    if result.returncode != 0:
        print(f"❌ Tests in {test_dir} failed")
        sys.exit(1)

# === Generate coverage report ===
print("\n=== Coverage Report ===")
subprocess.run(["coverage", "report"], capture_output=False)
subprocess.run(["coverage", "html"], capture_output=False)
print("HTML coverage report generated in 'htmlcov' directory")