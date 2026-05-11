"""
Dependency Resolver — resolves nested defined terms.

Problem: definitions reference other defined terms.
  "Material Adverse Effect" means any event affecting the Business...
  "Business" means the operations of the Company and its Subsidiaries...
  "Subsidiaries" means all entities controlled by the Company...

A chunk using "Material Adverse Effect" needs all 3 definitions, not just 1.

This module:
  1. Builds a directed graph: term A → term B means "A's definition uses B"
  2. Topological sort to resolve in correct dependency order
  3. Returns a fully-resolved definition for each term (with nested expansions)
  4. Detects and handles circular dependencies
"""

import re
from collections import defaultdict, deque
from dataclasses import dataclass, field

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


@dataclass
class ResolvedDefinition:
    term: str
    raw_definition: str          # original text
    resolved_definition: str     # with nested terms expanded inline
    dependencies: list[str]      # direct deps (1-hop)
    all_dependencies: list[str]  # transitive deps
    depth: int                   # max nesting depth


def build_dependency_graph(
    defs: dict[str, str],
) -> dict[str, list[str]]:
    """
    Build adjacency list: term → [list of defined terms it references].

    defs: { term_lowercase: definition_text }
    """
    graph: dict[str, list[str]] = defaultdict(list)
    term_set = set(defs.keys())

    for term, definition in defs.items():
        def_lower = definition.lower()
        for other_term in term_set:
            if other_term == term:
                continue
            # Check if other_term appears in this definition
            # Use word-boundary matching to avoid false positives
            # e.g. "company" should not match inside "accompanies"
            pattern = r'\b' + re.escape(other_term) + r'(?:s|\'s|ies)?\b'
            if re.search(pattern, def_lower):
                graph[term].append(other_term)

    return dict(graph)


def detect_cycles(graph: dict[str, list[str]]) -> list[list[str]]:
    """
    Find all cycles in the dependency graph.
    Returns list of cycles (each cycle is a list of terms).
    """
    if HAS_NETWORKX:
        G = nx.DiGraph()
        for node, neighbors in graph.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        return list(nx.simple_cycles(G))

    # Fallback: DFS cycle detection
    visited = set()
    rec_stack = set()
    cycles = []

    def dfs(node, path):
        visited.add(node)
        rec_stack.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, path + [neighbor])
            elif neighbor in rec_stack:
                cycle_start = path.index(neighbor) if neighbor in path else 0
                cycles.append(path[cycle_start:] + [neighbor])
        rec_stack.discard(node)

    for node in graph:
        if node not in visited:
            dfs(node, [node])

    return cycles


def topological_sort(graph: dict[str, list[str]], all_terms: set[str]) -> list[str]:
    """
    Kahn's algorithm topological sort.
    Terms with no dependencies come first — resolve them first.
    Cycles are broken by removing the lowest-confidence edge (alphabetical fallback).
    """
    in_degree = defaultdict(int)
    adj = defaultdict(list)

    for term in all_terms:
        in_degree[term] = in_degree.get(term, 0)  # ensure exists

    for term, deps in graph.items():
        for dep in deps:
            adj[dep].append(term)
            in_degree[term] += 1

    queue = deque(sorted(t for t in all_terms if in_degree[t] == 0))
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Any terms not in order are part of cycles — append them at the end
    remaining = [t for t in all_terms if t not in set(order)]
    order.extend(sorted(remaining))

    return order


def resolve_definitions(
    defs: dict[str, str],
    max_depth: int = 3,
    max_expansion_length: int = 800,
) -> dict[str, ResolvedDefinition]:
    """
    Main entry point. Takes flat {term: definition} and returns
    fully-resolved definitions with nested term expansions.

    max_depth: how many levels of nesting to expand (3 is usually enough)
    max_expansion_length: cap on resolved definition length to avoid token bloat
    """
    if not defs:
        return {}

    # Build the graph
    graph = build_dependency_graph(defs)
    all_terms = set(defs.keys())
    sort_order = topological_sort(graph, all_terms)

    # Resolve in topological order (dependencies before dependents)
    resolved: dict[str, ResolvedDefinition] = {}

    for term in sort_order:
        if term not in defs:
            continue

        raw_def = defs[term]
        direct_deps = graph.get(term, [])

        # Compute transitive dependencies (BFS up to max_depth)
        all_deps = set()
        frontier = set(direct_deps)
        for _ in range(max_depth):
            next_frontier = set()
            for dep in frontier:
                if dep not in all_deps:
                    all_deps.add(dep)
                    next_frontier.update(graph.get(dep, []))
            frontier = next_frontier - all_deps
            if not frontier:
                break

        # Build resolved definition: append mini-glossary of dependencies
        if all_deps:
            dep_glossary_parts = []
            for dep in sorted(all_deps):
                if dep in resolved:
                    dep_text = resolved[dep].raw_definition
                elif dep in defs:
                    dep_text = defs[dep]
                else:
                    continue
                # Truncate long dependency definitions
                dep_text = dep_text[:200] + "..." if len(dep_text) > 200 else dep_text
                dep_glossary_parts.append(f'"{dep}": {dep_text}')

            if dep_glossary_parts:
                glossary = "; ".join(dep_glossary_parts)
                resolved_def = f"{raw_def} [where: {glossary}]"
            else:
                resolved_def = raw_def
        else:
            resolved_def = raw_def

        # Cap length
        if len(resolved_def) > max_expansion_length:
            resolved_def = resolved_def[:max_expansion_length] + "..."

        # Calculate depth
        depth = 0
        if all_deps:
            def _depth(t, visited=None):
                if visited is None:
                    visited = set()
                if t in visited or t not in graph:
                    return 0
                visited.add(t)
                deps = graph.get(t, [])
                if not deps:
                    return 0
                return 1 + max((_depth(d, visited.copy()) for d in deps), default=0)
            depth = _depth(term)

        resolved[term] = ResolvedDefinition(
            term=term,
            raw_definition=raw_def,
            resolved_definition=resolved_def,
            dependencies=direct_deps,
            all_dependencies=list(all_deps),
            depth=depth,
        )

    return resolved


def get_terms_in_text(text: str, known_terms: set[str]) -> list[str]:
    """
    Find which defined terms appear in a piece of text.
    Handles plurals and possessives.
    """
    text_lower = text.lower()
    found = []
    for term in known_terms:
        pattern = r'\b' + re.escape(term) + r"(?:s|'s|ies)?\b"
        if re.search(pattern, text_lower):
            found.append(term)
    return found
