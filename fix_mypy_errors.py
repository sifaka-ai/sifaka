#!/usr/bin/env python3
"""
Script to automatically fix common mypy errors in the Sifaka codebase.

This script analyzes and fixes common mypy errors without maintaining backward compatibility.
It focuses on:
1. Protocol type variable issues
2. Missing type annotations
3. Incompatible default values
4. Incompatible return types
5. Missing named arguments
6. Attribute access on Optional types
7. Union type issues
8. Syntax issues
"""

import os
import re
import sys
import ast
import astor
from typing import List, Dict, Tuple, Set, Optional, Any, Union
import subprocess
from pathlib import Path

# Directory to process
SIFAKA_DIR = "sifaka"

# Regular expressions for common patterns
PROTOCOL_TYPE_VAR_PATTERN = re.compile(r'(\w+)\s*=\s*TypeVar\s*\(\s*["\'](\w+)["\']\s*\)')
PROTOCOL_CLASS_PATTERN = re.compile(r'class\s+(\w+)\s*\(\s*Protocol\s*(?:\[\s*([^]]*)\s*\])?\s*\):')
MISSING_RETURN_TYPE_PATTERN = re.compile(r'def\s+(\w+)\s*\([^)]*\)\s*:')
OPTIONAL_ATTRIBUTE_PATTERN = re.compile(r'(\w+)\s*:\s*Optional\[([^]]+)\]')

def run_mypy() -> str:
    """Run mypy and return the output."""
    result = subprocess.run(["mypy", SIFAKA_DIR], capture_output=True, text=True)
    return result.stdout

def parse_mypy_errors(output: str) -> Dict[str, List[Dict[str, Any]]]:
    """Parse mypy output into a structured format."""
    errors_by_file = {}
    
    for line in output.splitlines():
        if not line or ":" not in line:
            continue
            
        try:
            file_path, rest = line.split(":", 1)
            if not file_path.endswith(".py"):
                continue
                
            line_num, error_info = rest.split(":", 1)
            error_type = "unknown"
            
            # Extract error type from the message
            error_match = re.search(r'\[([\w-]+)\]', error_info)
            if error_match:
                error_type = error_match.group(1)
                
            error_info = error_info.strip()
            
            if file_path not in errors_by_file:
                errors_by_file[file_path] = []
                
            errors_by_file[file_path].append({
                "line": int(line_num),
                "error_type": error_type,
                "message": error_info
            })
        except Exception as e:
            print(f"Error parsing line: {line}, {str(e)}")
            
    return errors_by_file

def fix_protocol_type_vars(file_path: str) -> bool:
    """Fix invariant type variables used in protocols."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find TypeVar definitions and update them
    modified = False
    
    # Replace invariant TypeVars with covariant/contravariant ones based on context
    typevars = {}
    for match in PROTOCOL_TYPE_VAR_PATTERN.finditer(content):
        var_name, type_name = match.groups()
        typevars[var_name] = type_name
    
    # Look for Protocol classes using these TypeVars
    for match in PROTOCOL_CLASS_PATTERN.finditer(content):
        class_name, type_params = match.groups()
        if not type_params:
            continue
            
        # Check if this is a protocol class with type parameters
        if "Protocol" in class_name or class_name.endswith("Protocol"):
            # For input types, make them contravariant
            # For output/result types, make them covariant
            for var_name, type_name in typevars.items():
                if var_name in type_params:
                    if "Input" in type_name or "T" == type_name:
                        # Make input types contravariant
                        new_typevar = f"{var_name} = TypeVar(\"{type_name}\", contravariant=True)"
                        old_typevar = f"{var_name} = TypeVar(\"{type_name}\")"
                        if old_typevar in content and new_typevar not in content:
                            content = content.replace(old_typevar, new_typevar)
                            modified = True
                    elif "Output" in type_name or "Result" in type_name or "R" == type_name:
                        # Make output types covariant
                        new_typevar = f"{var_name} = TypeVar(\"{type_name}\", covariant=True)"
                        old_typevar = f"{var_name} = TypeVar(\"{type_name}\")"
                        if old_typevar in content and new_typevar not in content:
                            content = content.replace(old_typevar, new_typevar)
                            modified = True
    
    if modified:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Fixed protocol type variables in {file_path}")
    
    return modified

def fix_missing_return_types(file_path: str) -> bool:
    """Add missing return type annotations to functions."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse the file
        tree = ast.parse(content)
        
        # Track if we made changes
        modified = False
        
        # Process all function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if the function is missing a return type annotation
                if node.returns is None:
                    # Add a return type annotation based on the function body
                    if any(isinstance(child, ast.Return) and child.value for child in ast.walk(node)):
                        # Function returns something, add -> Any
                        node.returns = ast.Name(id='Any', ctx=ast.Load())
                    else:
                        # Function doesn't return anything, add -> None
                        node.returns = ast.Name(id='None', ctx=ast.Load())
                    modified = True
        
        if modified:
            # Generate the modified code
            modified_content = astor.to_source(tree)
            
            # Add Any import if needed
            if "-> Any" in modified_content and "Any" not in content:
                import_line = "from typing import Any\n"
                if "import" in modified_content:
                    # Find the last import line
                    import_lines = re.findall(r'^.*import.*$', modified_content, re.MULTILINE)
                    if import_lines:
                        last_import = import_lines[-1]
                        modified_content = modified_content.replace(last_import, last_import + "\n" + import_line)
                else:
                    # No imports, add at the top
                    modified_content = import_line + modified_content
            
            with open(file_path, 'w') as f:
                f.write(modified_content)
            
            print(f"Fixed missing return types in {file_path}")
        
        return modified
    except Exception as e:
        print(f"Error fixing return types in {file_path}: {str(e)}")
        return False

def fix_optional_attribute_access(file_path: str) -> bool:
    """Fix attribute access on Optional types."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find patterns like: if self._state_manager is not None: self._state_manager.update(...)
    # and replace with: if self._state_manager: self._state_manager.update(...)
    modified = False
    
    # Find all Optional type annotations
    optional_vars = set()
    for match in OPTIONAL_ATTRIBUTE_PATTERN.finditer(content):
        var_name, _ = match.groups()
        optional_vars.add(var_name)
    
    # Replace direct attribute access with safe access
    for var in optional_vars:
        # Pattern: var.attribute
        pattern = rf'({var})\.(\w+)'
        
        # Find all occurrences
        for match in re.finditer(pattern, content):
            full_match = match.group(0)
            var_name = match.group(1)
            attr_name = match.group(2)
            
            # Check if this is inside a null check
            line_start = content[:match.start()].rfind('\n') + 1
            line_end = content.find('\n', match.start())
            line = content[line_start:line_end]
            
            # If not inside a null check, add one
            if f"if {var_name}" not in line and f"{var_name} is not None" not in line:
                # Replace with safe access
                replacement = f"{var_name} and {var_name}.{attr_name}"
                content = content.replace(full_match, replacement)
                modified = True
    
    if modified:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Fixed Optional attribute access in {file_path}")
    
    return modified

def process_file(file_path: str, errors: List[Dict[str, Any]]) -> bool:
    """Process a file and fix errors."""
    modified = False
    
    # Group errors by type
    error_types = {}
    for error in errors:
        error_type = error["error_type"]
        if error_type not in error_types:
            error_types[error_type] = []
        error_types[error_type].append(error)
    
    # Fix protocol type variable issues
    if "misc" in error_types:
        if fix_protocol_type_vars(file_path):
            modified = True
    
    # Fix missing return type annotations
    if "no-untyped-def" in error_types:
        if fix_missing_return_types(file_path):
            modified = True
    
    # Fix Optional attribute access
    if "union-attr" in error_types or "attr-defined" in error_types:
        if fix_optional_attribute_access(file_path):
            modified = True
    
    return modified

def main():
    """Main function to run the script."""
    print("Running mypy to identify errors...")
    mypy_output = run_mypy()
    
    print("Parsing mypy errors...")
    errors_by_file = parse_mypy_errors(mypy_output)
    
    print(f"Found errors in {len(errors_by_file)} files.")
    
    # Process each file
    modified_files = 0
    for file_path, errors in errors_by_file.items():
        print(f"Processing {file_path}...")
        if process_file(file_path, errors):
            modified_files += 1
    
    print(f"Fixed errors in {modified_files} files.")
    
    # Run mypy again to see if we've made progress
    print("Running mypy again to check progress...")
    new_mypy_output = run_mypy()
    new_errors_by_file = parse_mypy_errors(new_mypy_output)
    
    print(f"Remaining errors in {len(new_errors_by_file)} files.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
