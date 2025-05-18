#!/bin/bash
# Script to format all Python code in the repository

# Run isort to sort imports
isort --profile black --line-length 100 sifaka tests

# Run black to format code
black --line-length 100 sifaka tests

# Run autoflake to remove unused imports
autoflake --in-place --recursive --remove-all-unused-imports --remove-unused-variables sifaka tests

# Run ruff with fixes
ruff check --fix --line-length 100 sifaka tests

echo "Code formatting complete!"
