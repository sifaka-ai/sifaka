#!/bin/bash

# This script replaces the old files with the new ones

# Create directories if they don't exist
mkdir -p sifaka/improvers
mkdir -p tests/core

# Move new files to replace old ones
mv sifaka/chain.py.new sifaka/chain.py
mv sifaka/__init__.py.new sifaka/__init__.py
mv sifaka/models/__init__.py.new sifaka/models/__init__.py
mv sifaka/validators/__init__.py.new sifaka/validators/__init__.py
mv examples/basic_example.py.new examples/basic_example.py
mv tests/test_dependency_injection.py.new tests/test_dependency_injection.py

# Remove old registry files
rm -f sifaka/core/registry.py
rm -f sifaka/core/initialize_registry.py

echo "Files replaced successfully!"
