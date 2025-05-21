"""Utilities for handling C++ docstrings."""

import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

def insert_cpp_docstring(file_path: str, docstring: str, start_line: int) -> bool:
    """
    Insert a docstring into a C++ file.
    
    Args:
        file_path: The path to the C++ file
        docstring: The docstring to insert
        start_line: The line number where the docstring should be inserted (1-based)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read the original file
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Format the docstring
        docstring_lines = format_cpp_docstring(docstring)
        
        # Insert the docstring before the start line (convert to 0-based)
        start_idx = start_line - 1
        
        # Find the proper indentation from the line we're inserting before
        if start_idx < len(lines):
            indent = get_indentation(lines[start_idx])
        else:
            indent = ""
        
        # Indent all docstring lines except the first one
        docstring_lines = [docstring_lines[0]] + [indent + line for line in docstring_lines[1:]]
        
        # Insert the docstring lines at the proper position
        new_lines = lines[:start_idx] + docstring_lines + lines[start_idx:]
        
        # Write back to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
            
        return True
        
    except Exception as e:
        logger.error(f"Error inserting C++ docstring: {e}")
        return False

def format_cpp_docstring(docstring: str) -> List[str]:
    """
    Format a docstring for C++ style documentation.
    
    Args:
        docstring: The docstring to format
        
    Returns:
        A list of lines in C++ documentation format
    """
    # Remove any leading/trailing whitespace
    docstring = docstring.strip()
    
    # Split into lines and strip each line
    lines = [line.strip() for line in docstring.splitlines()]
    
    # Format in C++ style with /** */ markers
    result = ["/**\n"]
    for line in lines:
        # Add asterisk prefix for each line
        result.append(f" * {line}\n")
    result.append(" */\n")
    
    return result

def get_indentation(line: str) -> str:
    """
    Get the leading whitespace from a line of code.
    
    Args:
        line: The line to analyze
        
    Returns:
        The indentation string
    """
    return line[:len(line) - len(line.lstrip())]
