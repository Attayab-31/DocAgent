# Copyright (c) Meta Platforms, Inc. and affiliates
"""
Dependency analyzer module for building and processing dependency graphs 
between Python and C++ code components.
"""

from .ast_parser import CodeComponent, DependencyParser
try:
    from .cpp_parser import CppComponent, CppParser
except ImportError:
    CppComponent = None
    CppParser = None
from .topo_sort import topological_sort, resolve_cycles, build_graph_from_components, dependency_first_dfs

__all__ = [
    'CodeComponent',
    'CppComponent', 
    'DependencyParser',
    'CppParser',
    'topological_sort',
    'resolve_cycles',
    'build_graph_from_components',
    'dependency_first_dfs'
]