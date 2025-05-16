"""
Utility functions for dependency graph management and cycle detection.

This module provides functions for managing dependency graphs,
detecting cycles, and analyzing dependency relationships.
"""

from typing import Dict, List, Optional, Set, Tuple


def build_dependency_graph(dependencies: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Build a dependency graph from a dictionary of dependencies.

    Args:
        dependencies: Dictionary mapping dependency names to their dependencies

    Returns:
        Complete dependency graph including indirect dependencies
    """
    graph = {}

    # Initialize the graph with direct dependencies
    for name, deps in dependencies.items():
        if name not in graph:
            graph[name] = []
        for dep in deps:
            if dep not in graph:
                graph[dep] = []
            if dep not in graph[name]:
                graph[name].append(dep)

    return graph


def detect_cycles(graph: Dict[str, List[str]]) -> List[List[str]]:
    """
    Detect cycles in a dependency graph.

    Args:
        graph: Dependency graph

    Returns:
        List of cycles, where each cycle is a list of dependency names
    """
    cycles = []
    visited = set()
    path = []
    path_set = set()

    def dfs(node: str) -> None:
        if node in path_set:
            # Found a cycle
            cycle_start = path.index(node)
            cycles.append(path[cycle_start:] + [node])
            return

        if node in visited:
            return

        visited.add(node)
        path.append(node)
        path_set.add(node)

        for neighbor in graph.get(node, []):
            dfs(neighbor)

        path.pop()
        path_set.remove(node)

    for node in graph:
        if node not in visited:
            dfs(node)

    return cycles


def find_cycle_containing(graph: Dict[str, List[str]], name: str) -> Optional[List[str]]:
    """
    Find a cycle containing the given dependency name.

    Args:
        graph: Dependency graph
        name: Dependency name to find in a cycle

    Returns:
        Cycle containing the dependency, or None if no cycle is found
    """
    if name not in graph:
        return None

    visited = set()
    path = []
    path_set = set()

    def dfs(node: str) -> Optional[List[str]]:
        if node in path_set:
            # Found a cycle
            cycle_start = path.index(node)
            cycle = path[cycle_start:] + [node]
            if name in cycle:
                return cycle
            return None

        if node in visited:
            return None

        visited.add(node)
        path.append(node)
        path_set.add(node)

        for neighbor in graph.get(node, []):
            cycle = dfs(neighbor)
            if cycle:
                return cycle

        path.pop()
        path_set.remove(node)
        return None

    return dfs(name)


def topological_sort(graph: Dict[str, List[str]]) -> List[str]:
    """
    Perform a topological sort on a dependency graph.

    Args:
        graph: Dependency graph

    Returns:
        List of dependency names in topological order

    Raises:
        ValueError: If the graph contains a cycle
    """
    if detect_cycles(graph):
        raise ValueError("Cannot perform topological sort on a graph with cycles")

    result = []
    visited = set()
    temp_mark = set()

    def visit(node: str) -> None:
        if node in temp_mark:
            # This should not happen if we've already checked for cycles
            raise ValueError(f"Cycle detected involving {node}")

        if node in visited:
            return

        temp_mark.add(node)

        for neighbor in graph.get(node, []):
            visit(neighbor)

        temp_mark.remove(node)
        visited.add(node)
        result.append(node)

    for node in graph:
        if node not in visited:
            visit(node)

    return list(reversed(result))


def analyze_dependencies(graph: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
    """
    Analyze dependencies to find relationship depths.

    Args:
        graph: Dependency graph

    Returns:
        Dictionary mapping each dependency to its dependencies and their depths
    """
    analysis = {}

    for node in graph:
        if node not in analysis:
            analysis[node] = {}

        # BFS to find all dependencies and their depths
        queue = [(dep, 1) for dep in graph.get(node, [])]
        visited = set(graph.get(node, []))

        while queue:
            dep, depth = queue.pop(0)
            analysis[node][dep] = depth

            for sub_dep in graph.get(dep, []):
                if sub_dep not in visited:
                    visited.add(sub_dep)
                    queue.append((sub_dep, depth + 1))

    return analysis


def find_all_dependents(graph: Dict[str, List[str]], name: str) -> Set[str]:
    """
    Find all dependents of a given dependency.

    Args:
        graph: Dependency graph
        name: Dependency name

    Returns:
        Set of all dependencies that depend on the given dependency
    """
    # Invert the graph
    inverted_graph = {}
    for node, deps in graph.items():
        for dep in deps:
            if dep not in inverted_graph:
                inverted_graph[dep] = []
            inverted_graph[dep].append(node)

    # BFS to find all dependents
    dependents = set()
    queue = inverted_graph.get(name, [])
    visited = set(queue)

    while queue:
        dependent = queue.pop(0)
        dependents.add(dependent)

        for parent in inverted_graph.get(dependent, []):
            if parent not in visited:
                visited.add(parent)
                queue.append(parent)

    return dependents


def find_all_dependencies(graph: Dict[str, List[str]], name: str) -> Set[str]:
    """
    Find all dependencies of a given dependency.

    Args:
        graph: Dependency graph
        name: Dependency name

    Returns:
        Set of all dependencies that the given dependency depends on
    """
    # BFS to find all dependencies
    dependencies = set()
    queue = graph.get(name, [])
    visited = set(queue)

    while queue:
        dependency = queue.pop(0)
        dependencies.add(dependency)

        for child in graph.get(dependency, []):
            if child not in visited:
                visited.add(child)
                queue.append(child)

    return dependencies
