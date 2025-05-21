"""
C++ code parser that extracts dependency information between code components.

This module identifies includes and references between C++ code components (functions, classes, methods)
and builds a dependency graph for topological sorting.
"""

import os
import re
import json
import sys
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Union
from pathlib import Path

logger = logging.getLogger(__name__)

# Initialize clang
CLANG_AVAILABLE = False
CursorKind = None
TokenKind = None
Index = None

try:
    import clang.cindex
    LLVM_PATH = r"C:\Program Files\LLVM"
    LIBCLANG_PATH = os.path.join(LLVM_PATH, "bin", "libclang.dll")

    if os.path.exists(LIBCLANG_PATH):
        clang.cindex.Config.set_library_file(LIBCLANG_PATH)
        from clang.cindex import CursorKind, TokenKind, Index
        CLANG_AVAILABLE = True
        logger.info("Successfully initialized libclang")
    else:
        logger.error(f"libclang.dll not found at {LIBCLANG_PATH}")
except ImportError:
    logger.error("clang module not found. Please install: pip install clang")
    CLANG_AVAILABLE = False
except Exception as e:
    logger.error(f"Error initializing clang: {e}")
    CLANG_AVAILABLE = False

def _extract_docstring(cursor) -> Optional[str]:
    """Extract docstring from comments before the declaration."""
    if not CLANG_AVAILABLE:
        return None
        
    try:
        tokens = list(cursor.get_tokens())
        if not tokens:
            return None

        # Look for comments before the declaration
        comments = []
        for token in tokens:
            if token.kind == TokenKind.COMMENT:
                comment_text = token.spelling
                # Clean up comment markers
                comment_text = comment_text.replace('/*', '').replace('*/', '').replace('//', '')
                comment_text = '\n'.join(line.strip() for line in comment_text.split('\n'))
                comments.append(comment_text)
            else:
                break

        return '\n'.join(comments) if comments else None
    except Exception as e:
        logger.error(f"Error extracting docstring: {e}")
        return None

@dataclass
class CppComponent:
    """Represents a C++ code component (function, class, or method)."""
    id: str
    component_type: str  # 'function', 'class', or 'method'  
    file_path: str
    relative_path: str
    depends_on: Set[str] = field(default_factory=set)
    start_line: int = 0
    end_line: int = 0
    has_docstring: bool = False
    docstring: str = ""
    language: str = "cpp"
    source_code: str = ""  # Added source_code attribute

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            'id': self.id,
            'component_type': self.component_type,
            'file_path': self.file_path,
            'relative_path': self.relative_path,
            'depends_on': list(self.depends_on),
            'start_line': self.start_line,
            'end_line': self.end_line,
            'has_docstring': self.has_docstring,
            'docstring': self.docstring,
            'language': self.language,
            'source_code': self.source_code
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CppComponent':
        """Create a CppComponent from a dictionary."""
        component = cls(
            id=data['id'],
            component_type=data['component_type'],
            file_path=data['file_path'],
            relative_path=data['relative_path']
        )
        component.depends_on = set(data['depends_on'])
        component.start_line = data['start_line']
        component.end_line = data['end_line']
        component.has_docstring = data['has_docstring']
        component.docstring = data['docstring']
        component.language = data.get('language', 'cpp')
        component.source_code = data.get('source_code', '')  # Added source_code handling
        return component

class CppParser:
    """Parser for C++ files that uses clang to build an AST and extract dependency information."""

    def __init__(self, repo_path: str):
        """Initialize the C++ parser."""
        self.repo_path = os.path.abspath(repo_path)
        self.components: Dict[str, CppComponent] = {}
        logger.info("Initialized C++ parser with repo path: %s", self.repo_path)

    def parse_file(self, file_path: str) -> Dict[str, CppComponent]:
        """Parse a C++ file to collect components and their dependencies."""
        if not CLANG_AVAILABLE:
            logger.error("clang module not available")
            return {}

        try:
            # Get relative path for component IDs
            relative_path = os.path.relpath(file_path, self.repo_path)
            module_path = relative_path.replace('/', '.').replace('\\', '.').rsplit('.', 1)[0]
            logger.info("Parsing C++ file: %s (module path: %s)", relative_path, module_path)

            # Parse the file with clang
            logger.debug("Creating clang Index")
            index = Index.create()
            logger.debug("Parsing translation unit")
            args = ['-x', 'c++', '-std=c++11']  # Add C++ flags
            translation_unit = index.parse(file_path, args=args)

            if not translation_unit:
                logger.error("Failed to parse translation unit")
                return {}

            logger.debug("Successfully parsed translation unit")

            # Reset components dict for this file
            self.components = {}
            
            # Visit all nodes in the AST
            logger.debug("Starting AST traversal with cursor kind: %s", str(translation_unit.cursor.kind))
            self._visit_ast(translation_unit.cursor, file_path, relative_path, module_path)
            
            logger.info("Successfully parsed %d components from %s", len(self.components), relative_path)
            return self.components
            
        except Exception as e:
            logger.error("Error parsing C++ file %s: %s", file_path, str(e))
            return {}
            
    def _visit_ast(self, cursor, file_path: str, relative_path: str, module_path: str) -> None:
        """Process each node in the clang AST."""
        try:
            # Special handling for translation unit cursor
            if cursor.kind == CursorKind.TRANSLATION_UNIT:
                logger.debug("Processing translation unit cursor")
                for child in cursor.get_children():
                    self._visit_ast(child, file_path, relative_path, module_path)
                return
            
            # For non-translation unit cursors, check file matching
            cursor_file = cursor.location.file
            if not cursor_file:
                logger.debug("Cursor has no associated file")
                return
                
            # Normalize paths for comparison
            cursor_path = os.path.abspath(cursor_file.name)
            file_path = os.path.abspath(file_path)
            if cursor_path != file_path:
                logger.debug("Skipping cursor from different file: %s != %s", cursor_path, file_path)
                return

            logger.debug("Processing cursor - Kind: %s, Spelling: '%s', Location: line %d", 
                       str(cursor.kind), cursor.spelling, cursor.location.line)

            # Handle different types of declarations
            if cursor.is_definition():
                if cursor.kind == CursorKind.FUNCTION_DECL:
                    logger.debug("Found function definition: %s", cursor.spelling)
                    component = self._process_function(cursor, file_path, relative_path, module_path)
                    if component:
                        logger.debug("Adding function component: %s", component.id)
                        self.components[component.id] = component

                elif cursor.kind == CursorKind.CLASS_DECL:
                    logger.debug("Found class definition: %s", cursor.spelling)
                    component = self._process_class(cursor, file_path, relative_path, module_path)
                    if component:
                        logger.debug("Adding class component: %s", component.id)
                        self.components[component.id] = component

                elif cursor.kind == CursorKind.CXX_METHOD:
                    logger.debug("Found method definition: %s", cursor.spelling)
                    component = self._process_method(cursor, file_path, relative_path, module_path)
                    if component:
                        logger.debug("Adding method component: %s", component.id)
                        self.components[component.id] = component

            # Recursively visit children
            for child in cursor.get_children():
                self._visit_ast(child, file_path, relative_path, module_path)

        except Exception as e:
            logger.error("Error visiting AST node: %s", str(e))

    def _get_source_code(self, cursor) -> str:
        """Extract source code for a cursor using its source range."""
        try:
            file = cursor.extent.start.file
            if not file:
                return ""
            
            # Get the start and end line numbers
            start_line = cursor.extent.start.line 
            end_line = cursor.extent.end.line
            
            # Read those lines from the file
            with open(file.name, 'r', encoding='utf-8') as f:
                content = f.readlines()
                
            # Extract the lines for this cursor (convert from 1-based to 0-based indexing)
            source_lines = content[start_line-1:end_line]
            return ''.join(source_lines)
            
        except Exception as e:
            logger.error(f"Error extracting source code: {e}")
            return ""

    def _process_function(self, cursor, file_path: str, relative_path: str, module_path: str) -> Optional[CppComponent]:
        """Process a C++ function declaration."""
        try:
            start_line = cursor.extent.start.line
            end_line = cursor.extent.end.line

            # Skip declarations without bodies
            if start_line == end_line:
                logger.debug("Skipping function without body: %s", cursor.spelling)
                return None

            function_name = cursor.spelling
            logger.info("Processing function: %s (lines %d-%d)", function_name, start_line, end_line)

            # Create component ID
            component_id = f"{module_path}.{function_name}"

            # Extract docstring from comments
            docstring = _extract_docstring(cursor)
            has_docstring = bool(docstring)

            # Extract source code
            source_code = self._get_source_code(cursor)

            # Create component
            component = CppComponent(
                id=component_id,
                component_type="function",
                file_path=file_path,
                relative_path=relative_path,
                depends_on=set(),
                start_line=start_line,
                end_line=end_line,
                has_docstring=has_docstring,
                docstring=docstring or "",
                language="cpp",
                source_code=source_code
            )

            logger.info("Added C++ function component: %s", component_id)
            return component

        except Exception as e:
            logger.error("Error processing function %s: %s", cursor.spelling, str(e))
            return None

    def _process_class(self, cursor, file_path: str, relative_path: str, module_path: str) -> Optional[CppComponent]:
        """Process a C++ class declaration."""
        try:
            class_name = cursor.spelling
            start_line = cursor.extent.start.line
            end_line = cursor.extent.end.line

            # Skip forward declarations
            if start_line == end_line:
                logger.debug("Skipping class forward declaration: %s", class_name)
                return None

            logger.info("Processing class: %s (lines %d-%d)", class_name, start_line, end_line)

            # Create component ID
            component_id = f"{module_path}.{class_name}"

            # Extract docstring from comments
            docstring = _extract_docstring(cursor)
            has_docstring = bool(docstring)

            # Extract source code
            source_code = self._get_source_code(cursor)

            # Create component
            component = CppComponent(
                id=component_id,
                component_type="class",
                file_path=file_path,
                relative_path=relative_path,
                depends_on=set(),
                start_line=start_line,
                end_line=end_line,
                has_docstring=has_docstring,
                docstring=docstring or "",
                language="cpp",
                source_code=source_code
            )

            logger.info("Added C++ class component: %s", component_id)
            return component

        except Exception as e:
            logger.error("Error processing class %s: %s", cursor.spelling, str(e))
            return None

    def _process_method(self, cursor, file_path: str, relative_path: str, module_path: str) -> Optional[CppComponent]:
        """Process a C++ method declaration."""
        try:
            start_line = cursor.extent.start.line
            end_line = cursor.extent.end.line

            # Skip method declarations without bodies
            if start_line == end_line:
                logger.debug("Skipping method without body: %s", cursor.spelling)
                return None

            method_name = cursor.spelling
            class_name = cursor.semantic_parent.spelling
            logger.info("Processing method: %s::%s (lines %d-%d)", class_name, method_name, start_line, end_line)

            # Create component ID
            component_id = f"{module_path}.{class_name}.{method_name}"

            # Extract docstring from comments
            docstring = _extract_docstring(cursor)
            has_docstring = bool(docstring)

            # Extract source code
            source_code = self._get_source_code(cursor)

            # Create component
            component = CppComponent(
                id=component_id,
                component_type="method",
                file_path=file_path,
                relative_path=relative_path,
                depends_on=set(),
                start_line=start_line,
                end_line=end_line,
                has_docstring=has_docstring,
                docstring=docstring or "",
                language="cpp",
                source_code=source_code
            )

            logger.info("Added C++ method component: %s", component_id)
            return component

        except Exception as e:
            logger.error("Error processing method %s: %s", cursor.spelling, str(e))
            return None

    def _resolve_component_dependencies(self, component: CppComponent) -> None:
        """
        Resolve dependencies for a C++ component using clang AST.
        
        Args:
            component: The C++ component to analyze for dependencies
        """
        try:
            import clang.cindex as cindex
            clang.cindex.Config.set_library_file(r"C:\Program Files\LLVM\bin\libclang.dll")

            # Parse the file
            index = cindex.Index.create()
            translation_unit = index.parse(component.file_path)
            
            # Find the cursor for this component
            cursor = self._find_component_cursor(translation_unit.cursor, component)
            if not cursor:
                return
            
            # Collect dependencies
            dependencies = set()
            
            # Process references in the cursor
            for child in cursor.walk_preorder():
                if child.referenced and child.referenced.location.file:
                    ref_path = child.referenced.location.file.name
                    if ref_path.startswith(self.repo_path):
                        # Convert file path to module path
                        relative_path = os.path.relpath(ref_path, self.repo_path)
                        module_path = relative_path.replace('/', '.').replace('\\', '.')
                        
                        # Get referenced component name
                        ref_name = child.referenced.spelling
                        if child.referenced.semantic_parent:
                            parent_name = child.referenced.semantic_parent.spelling
                            if parent_name:
                                ref_name = f"{parent_name}.{ref_name}"
                        
                        dependency = f"{module_path}.{ref_name}"
                        dependencies.add(dependency)
            
            # Update component dependencies
            component.depends_on.update(dependencies)
            
        except Exception as e:
            logger.error(f"Error resolving dependencies for {component.id}: {e}")

    def _find_component_cursor(self, cursor, component: CppComponent):
        """Find the cursor corresponding to a component in the AST."""
        if cursor.location.file and cursor.location.file.name == component.file_path:
            if cursor.extent.start.line == component.start_line:
                return cursor
                
            # Check if this is the component we're looking for
            if component.component_type == "class" and cursor.kind == CursorKind.CLASS_DECL:
                if cursor.spelling == component.id.split('.')[-1]:
                    return cursor
            elif component.component_type == "function" and cursor.kind == CursorKind.FUNCTION_DECL:
                if cursor.spelling == component.id.split('.')[-1]:
                    return cursor
            elif component.component_type == "method" and cursor.kind == CursorKind.CXX_METHOD:
                method_name = component.id.split('.')[-1]
                class_name = component.id.split('.')[-2]
                if cursor.spelling == method_name and cursor.semantic_parent.spelling == class_name:
                    return cursor
        
        # Recurse through children
        for child in cursor.get_children():
            result = self._find_component_cursor(child, component)
            if result:
                return result
        
        return None
