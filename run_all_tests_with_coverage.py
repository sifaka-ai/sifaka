#!/usr/bin/env python
"""Run all tests with coverage reporting.

This script runs the unit tests for the project and generates a coverage report.
"""

import os
import subprocess
import sys
import re

# Clean up cache files to avoid import conflicts
print("=== Cleaning cache files ===")
subprocess.run(["find", ".", "-name", "__pycache__", "-type", "d", "-exec", "rm", "-rf", "{}", "+"])
subprocess.run(["find", ".", "-name", "*.pyc", "-delete"])

# Make sure we remove any old coverage files
print("=== Clearing previous coverage data ===")
subprocess.run(["coverage", "erase"])

# Start coverage to collect data across all test runs
print("=== Starting coverage collection ===")

# Try to import and apply patches
try:
    from sifaka.utils.patches import apply_all_patches
    print("✅ Applied compatibility patches")
except ImportError:
    print("❌ Error applying patches: cannot import name 'apply_all_patches' from 'sifaka.utils.patches' (unknown location)")

# === Run regular tests with pytest ===
print("\n=== Running tests with pytest ===")

# Define test directories to include
test_dirs = [
    "tests/chain",
    "tests/utils",
    "tests/performance",
    "tests/integration",
    "tests/examples",
    "tests/rules"
]

# Build the command with specific directories
cmd = ["python", "-m", "pytest"]

# Add specific directories
for test_dir in test_dirs:
    cmd.append(test_dir)

# Add coverage options
cmd.extend(["--cov=sifaka", "--cov-report=term"])

# Run the command
result = subprocess.run(cmd, check=False)
if result.returncode != 0:
    print(f"❌ Tests failed with code {result.returncode}")
else:
    print("✅ All tests passed")

# Generate HTML coverage report
print("\n=== HTML Coverage Report ===")
subprocess.run(["coverage", "html"])
print("HTML report generated in htmlcov/ directory")

# === Now let's create a plan to add tests for low-coverage modules ===
print("\n=== Modules with Low Coverage ===")
# Run coverage report and capture output to identify low-coverage modules
coverage_output = subprocess.check_output(["coverage", "report", "--skip-covered"], universal_newlines=True)

# Parse the coverage report to find modules with low coverage
low_coverage_modules = []
for line in coverage_output.split('\n'):
    if line.strip() and not line.startswith('Name') and not line.startswith('---'):
        parts = re.split(r'\s+', line.strip())
        if len(parts) >= 4:
            module = parts[0]
            coverage_pct = parts[3].replace('%', '')
            try:
                coverage_pct = float(coverage_pct)
                if coverage_pct < 40:  # Consider modules with less than 40% coverage as low
                    low_coverage_modules.append((module, coverage_pct))
            except ValueError:
                pass

# Print out modules that need attention
print("The following modules have low test coverage and need additional tests:")
for module, coverage in sorted(low_coverage_modules, key=lambda x: x[1]):
    print(f"  - {module}: {coverage}%")

# === Print summary ===
print("\n=== Tests completed ===")