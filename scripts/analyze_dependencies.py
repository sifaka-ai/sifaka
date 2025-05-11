#!/usr/bin/env python3
"""
Script to analyze dependencies in the Sifaka codebase.

This script scans the Sifaka codebase for import statements and builds a dependency graph.
It then identifies potential circular dependencies and outputs a report.

Usage:
    python scripts/analyze_dependencies.py

Output:
    - A report of potential circular dependencies
    - A visualization of the dependency graph (if graphviz is installed)
"""

import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# Add the parent directory to the path so we can import Sifaka modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Regular expression to match import statements
IMPORT_RE = re.compile(r"^\s*(?:from\s+([.\w]+)\s+import|import\s+([.\w]+))")
MULTILINE_IMPORT_RE = re.compile(r"^\s*from\s+([.\w]+)\s+import\s+\(")

# Modules to ignore in the analysis
IGNORE_MODULES = {
    "typing",
    "os",
    "sys",
    "time",
    "json",
    "re",
    "datetime",
    "logging",
    "pathlib",
    "collections",
    "functools",
    "itertools",
    "abc",
    "enum",
    "inspect",
    "asyncio",
    "contextlib",
    "copy",
    "dataclasses",
    "importlib",
    "math",
    "random",
    "tempfile",
    "uuid",
    "warnings",
    "zlib",
    "base64",
    "hashlib",
    "hmac",
    "io",
    "pickle",
    "shutil",
    "socket",
    "ssl",
    "string",
    "struct",
    "subprocess",
    "threading",
    "traceback",
    "urllib",
    "weakref",
    "xml",
    "yaml",
    "pytest",
    "unittest",
    "mock",
    "requests",
    "numpy",
    "pandas",
    "pydantic",
}


def find_python_files(root_dir: str) -> List[str]:
    """Find all Python files in the given directory and its subdirectories."""
    python_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                python_files.append(os.path.join(dirpath, filename))
    return python_files


def extract_imports(file_path: str) -> List[str]:
    """Extract import statements from a Python file."""
    imports = []
    in_multiline_import = False
    multiline_import_module = None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # Skip comments
            if line.strip().startswith("#"):
                continue

            # Check for multiline import start
            if not in_multiline_import:
                multiline_match = MULTILINE_IMPORT_RE.match(line)
                if multiline_match:
                    in_multiline_import = True
                    multiline_import_module = multiline_match.group(1)
                    continue

            # Check for multiline import end
            if in_multiline_import:
                if ")" in line:
                    in_multiline_import = False
                    multiline_import_module = None
                continue

            # Check for regular import
            match = IMPORT_RE.match(line)
            if match:
                module = match.group(1) or match.group(2)
                # Skip standard library and third-party modules
                if not any(module.startswith(ignore) for ignore in IGNORE_MODULES):
                    imports.append(module)

    return imports


def build_dependency_graph(python_files: List[str]) -> Dict[str, Set[str]]:
    """Build a dependency graph from Python files."""
    graph = defaultdict(set)
    file_to_module = {}

    # Map file paths to module names
    for file_path in python_files:
        rel_path = os.path.relpath(file_path, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        module_path = rel_path.replace("/", ".").replace("\\", ".").replace(".py", "")
        file_to_module[file_path] = module_path

    # Build the graph
    for file_path in python_files:
        module = file_to_module[file_path]
        imports = extract_imports(file_path)
        for imported_module in imports:
            # Only add dependencies within the codebase
            if any(imported_module == m or m.startswith(imported_module + ".") for m in file_to_module.values()):
                graph[module].add(imported_module)

    return graph


def find_circular_dependencies(graph: Dict[str, Set[str]]) -> List[List[str]]:
    """Find circular dependencies in the dependency graph."""
    circular_deps = []
    visited = set()
    path = []

    def dfs(node):
        if node in path:
            # Found a cycle
            cycle = path[path.index(node):] + [node]
            circular_deps.append(cycle)
            return
        if node in visited:
            return

        visited.add(node)
        path.append(node)

        for neighbor in graph.get(node, []):
            dfs(neighbor)

        path.pop()

    for node in graph:
        if node not in visited:
            dfs(node)

    return circular_deps


def print_dependency_report(graph: Dict[str, Set[str]], circular_deps: List[List[str]]):
    """Print a report of the dependency graph and circular dependencies."""
    print("=== Dependency Analysis Report ===")
    print(f"Total modules: {len(graph)}")
    print(f"Total dependencies: {sum(len(deps) for deps in graph.values())}")
    print(f"Circular dependencies: {len(circular_deps)}")
    print()

    if circular_deps:
        print("=== Circular Dependencies ===")
        for i, cycle in enumerate(circular_deps, 1):
            print(f"Cycle {i}: {' -> '.join(cycle)}")
        print()

    # Print modules with the most dependencies
    print("=== Modules with Most Dependencies ===")
    sorted_modules = sorted(graph.items(), key=lambda x: len(x[1]), reverse=True)
    for module, deps in sorted_modules[:10]:
        print(f"{module}: {len(deps)} dependencies")
    print()

    # Print modules with the most dependents
    print("=== Most Depended-Upon Modules ===")
    dependents = defaultdict(set)
    for module, deps in graph.items():
        for dep in deps:
            dependents[dep].add(module)
    sorted_dependents = sorted(dependents.items(), key=lambda x: len(x[1]), reverse=True)
    for module, deps in sorted_dependents[:10]:
        print(f"{module}: {len(deps)} dependents")


def main():
    """Main function."""
    # Get the Sifaka root directory
    sifaka_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sifaka_package_dir = os.path.join(sifaka_dir, "sifaka")

    # Find all Python files
    python_files = find_python_files(sifaka_package_dir)
    print(f"Found {len(python_files)} Python files")

    # Build the dependency graph
    graph = build_dependency_graph(python_files)
    print(f"Built dependency graph with {len(graph)} modules")

    # Find circular dependencies
    circular_deps = find_circular_dependencies(graph)
    print(f"Found {len(circular_deps)} circular dependencies")

    # Print the report
    print_dependency_report(graph, circular_deps)

    # Save the report to a file
    report_path = os.path.join(sifaka_dir, "dependency_analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Dependency Analysis Report\n\n")
        f.write(f"Total modules: {len(graph)}\n")
        f.write(f"Total dependencies: {sum(len(deps) for deps in graph.values())}\n")
        f.write(f"Circular dependencies: {len(circular_deps)}\n\n")

        if circular_deps:
            f.write("## Circular Dependencies\n\n")
            for i, cycle in enumerate(circular_deps, 1):
                f.write(f"### Cycle {i}\n")
                f.write(f"```\n{' -> '.join(cycle)}\n```\n\n")

        f.write("## Modules with Most Dependencies\n\n")
        sorted_modules = sorted(graph.items(), key=lambda x: len(x[1]), reverse=True)
        for module, deps in sorted_modules[:20]:
            f.write(f"- **{module}**: {len(deps)} dependencies\n")
            for dep in sorted(deps)[:10]:  # Show only the first 10 dependencies
                f.write(f"  - {dep}\n")
            if len(deps) > 10:
                f.write(f"  - ... and {len(deps) - 10} more\n")
            f.write("\n")

        f.write("## Most Depended-Upon Modules\n\n")
        dependents = defaultdict(set)
        for module, deps in graph.items():
            for dep in deps:
                dependents[dep].add(module)
        sorted_dependents = sorted(dependents.items(), key=lambda x: len(x[1]), reverse=True)
        for module, deps in sorted_dependents[:20]:
            f.write(f"- **{module}**: {len(deps)} dependents\n")

    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
