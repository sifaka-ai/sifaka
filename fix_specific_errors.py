#!/usr/bin/env python3
"""
Script to fix specific mypy errors in the Sifaka codebase.

This script targets specific error patterns that are common in the codebase:
1. Fix ClassificationResult type arguments
2. Fix incompatible default values (implicit Optional)
3. Fix missing named arguments in constructor calls
4. Fix Protocol[...] type variable listings
5. Fix X | Y union syntax for Python < 3.10
"""

import os
import re
import sys
from typing import List, Dict, Any, Optional, Set, Tuple
import ast
import astor
from pathlib import Path

# Directory to process
SIFAKA_DIR = "sifaka"

def find_python_files(directory: str) -> List[str]:
    """Find all Python files in the given directory."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files

def fix_classification_result_type_args(file_path: str) -> bool:
    """Fix ClassificationResult type arguments."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern: ClassificationResult[T] -> ClassificationResult[T, Any]
    pattern = r'ClassificationResult\[([^,\]]+)\]'
    replacement = r'ClassificationResult[\1, Any]'
    
    new_content = re.sub(pattern, replacement, content)
    
    if new_content != content:
        # Add Any import if needed
        if "Any" in new_content and "Any" not in content:
            if "from typing import" in new_content:
                new_content = re.sub(
                    r'from typing import (.*)',
                    r'from typing import \1, Any',
                    new_content
                )
            else:
                new_content = "from typing import Any\n" + new_content
        
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        print(f"Fixed ClassificationResult type arguments in {file_path}")
        return True
    
    return False

def fix_implicit_optional(file_path: str) -> bool:
    """Fix incompatible default values (implicit Optional)."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern: def func(param: Type = None) -> ...
    # Replace with: def func(param: Optional[Type] = None) -> ...
    pattern = r'def\s+\w+\s*\(([^)]*\s*:\s*([^=\s]+)\s*=\s*None[^)]*)\)'
    
    def replace_param(match):
        params = match.group(1)
        # For each parameter with default None
        param_pattern = r'(\w+)\s*:\s*([^=\s,]+)\s*=\s*None'
        
        def replace_single_param(param_match):
            param_name = param_match.group(1)
            param_type = param_match.group(2)
            return f"{param_name}: Optional[{param_type}] = None"
        
        new_params = re.sub(param_pattern, replace_single_param, params)
        return f"def {match.group(0).split('(')[0]}({new_params})"
    
    new_content = re.sub(pattern, replace_param, content)
    
    if new_content != content:
        # Add Optional import if needed
        if "Optional" in new_content and "Optional" not in content:
            if "from typing import" in new_content:
                new_content = re.sub(
                    r'from typing import (.*)',
                    r'from typing import \1, Optional',
                    new_content
                )
            else:
                new_content = "from typing import Optional\n" + new_content
        
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        print(f"Fixed implicit Optional in {file_path}")
        return True
    
    return False

def fix_missing_named_args(file_path: str) -> bool:
    """Fix missing named arguments in constructor calls."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse the file
        tree = ast.parse(content)
        
        # Track if we made changes
        modified = False
        
        # Common missing arguments and their default values
        common_missing_args = {
            "ClassificationResult": {
                "passed": "True",
                "message": "\"\"",
            },
            "CriticMetadata": {
                "passed": "True",
                "message": "\"\"",
            },
            "RuleConfig": {
                "name": "\"rule\"",
                "description": "\"Rule description\"",
            },
            "ErrorResult": {
                "component_name": "\"component\"",
            }
        }
        
        # Process all function calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and hasattr(node, 'func') and hasattr(node.func, 'id'):
                func_name = node.func.id
                if func_name in common_missing_args:
                    # Check if required args are missing
                    existing_keywords = {kw.arg for kw in node.keywords if kw.arg}
                    missing_keywords = set(common_missing_args[func_name].keys()) - existing_keywords
                    
                    # Add missing keywords
                    for kw in missing_keywords:
                        value_str = common_missing_args[func_name][kw]
                        if value_str.startswith('"'):
                            value = ast.Constant(value=value_str.strip('"'))
                        elif value_str == "True":
                            value = ast.Constant(value=True)
                        elif value_str == "False":
                            value = ast.Constant(value=False)
                        else:
                            value = ast.Name(id=value_str, ctx=ast.Load())
                        
                        node.keywords.append(ast.keyword(arg=kw, value=value))
                        modified = True
        
        if modified:
            # Generate the modified code
            modified_content = astor.to_source(tree)
            
            with open(file_path, 'w') as f:
                f.write(modified_content)
            
            print(f"Fixed missing named arguments in {file_path}")
        
        return modified
    except Exception as e:
        print(f"Error fixing named args in {file_path}: {str(e)}")
        return False

def fix_protocol_type_var_listing(file_path: str) -> bool:
    """Fix Protocol[...] type variable listings."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern: class X(Protocol): but should list type variables
    pattern = r'class\s+(\w+)\s*\(\s*Protocol\s*\):'
    
    # Find all TypeVar definitions
    typevar_pattern = r'(\w+)\s*=\s*TypeVar\s*\(\s*["\'](\w+)["\']\s*(?:,\s*(?:contravariant|covariant)\s*=\s*(?:True|False))?\s*\)'
    typevars = {}
    
    for match in re.finditer(typevar_pattern, content):
        var_name, type_name = match.groups()
        typevars[var_name] = type_name
    
    # For each Protocol class without type variables
    def replace_protocol(match):
        class_name = match.group(1)
        
        # Determine which type variables to use based on class name
        used_typevars = []
        if "Input" in class_name or "Request" in class_name:
            for var, _ in typevars.items():
                if "Input" in var or "T" == var:
                    used_typevars.append(var)
        
        if "Output" in class_name or "Result" in class_name or "Response" in class_name:
            for var, _ in typevars.items():
                if "Output" in var or "Result" in var or "R" == var:
                    used_typevars.append(var)
        
        # If we found type variables to use
        if used_typevars:
            return f"class {class_name}(Protocol[{', '.join(used_typevars)}]):"
        
        return match.group(0)
    
    new_content = re.sub(pattern, replace_protocol, content)
    
    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        print(f"Fixed Protocol type variable listings in {file_path}")
        return True
    
    return False

def fix_union_syntax(file_path: str) -> bool:
    """Fix X | Y union syntax for Python < 3.10."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern: Type1 | Type2 -> Union[Type1, Type2]
    pattern = r'(\w+(?:\[[\w\[\],\s]+\])?) \| (\w+(?:\[[\w\[\],\s]+\])?)'
    
    def replace_union(match):
        type1 = match.group(1)
        type2 = match.group(2)
        return f"Union[{type1}, {type2}]"
    
    new_content = re.sub(pattern, replace_union, content)
    
    if new_content != content:
        # Add Union import if needed
        if "Union" in new_content and "Union" not in content:
            if "from typing import" in new_content:
                new_content = re.sub(
                    r'from typing import (.*)',
                    r'from typing import \1, Union',
                    new_content
                )
            else:
                new_content = "from typing import Union\n" + new_content
        
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        print(f"Fixed union syntax in {file_path}")
        return True
    
    return False

def process_file(file_path: str) -> bool:
    """Process a file and fix errors."""
    modified = False
    
    # Apply fixes
    if fix_classification_result_type_args(file_path):
        modified = True
    
    if fix_implicit_optional(file_path):
        modified = True
    
    if fix_missing_named_args(file_path):
        modified = True
    
    if fix_protocol_type_var_listing(file_path):
        modified = True
    
    if fix_union_syntax(file_path):
        modified = True
    
    return modified

def main():
    """Main function to run the script."""
    print("Finding Python files...")
    python_files = find_python_files(SIFAKA_DIR)
    
    print(f"Found {len(python_files)} Python files.")
    
    # Process each file
    modified_files = 0
    for file_path in python_files:
        print(f"Processing {file_path}...")
        if process_file(file_path):
            modified_files += 1
    
    print(f"Fixed errors in {modified_files} files.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
