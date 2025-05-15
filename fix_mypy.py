#!/usr/bin/env python
import sys
import subprocess
import re
from collections import defaultdict
import argparse


def parse_mypy_output(file_path):
    """
    Run mypy on the specified file and parse the output.
    Returns a dictionary of error types to lists of errors.
    """
    try:
        result = subprocess.run(
            ["mypy", "--no-error-summary", file_path],
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout
        if not output.strip():
            return {}

        errors = defaultdict(list)
        lines = output.strip().split("\n")

        for line in lines:
            # Extract the error type using regex
            error_match = re.search(r"\[([^]]+)\]$", line)
            if error_match:
                error_type = error_match.group(1)
                errors[error_type].append(line)

        return errors
    except Exception as e:
        print(f"Error running mypy: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Run mypy on a file and organize errors by type")
    parser.add_argument("file_path", type=str, help="Path to the file to analyze")
    args = parser.parse_args()

    errors = parse_mypy_output(args.file_path)

    if not errors:
        print(f"No mypy errors found in {args.file_path}")
        return

    print(f"Mypy errors found in {args.file_path}:")
    print("=" * 80)

    # Sort error types by frequency
    sorted_errors = sorted(errors.items(), key=lambda x: len(x[1]), reverse=True)

    for error_type, error_lines in sorted_errors:
        print(f"\n{error_type} ({len(error_lines)} errors):")
        print("-" * 80)
        for line in error_lines:
            print(line)


if __name__ == "__main__":
    main()
