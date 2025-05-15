#!/usr/bin/env python
"""
Script to fix the '_initialize_state undefined in superclass' error
by making classes inherit from InitializeStateMixin.
"""

import os
import sys
import re
from typing import List, Dict, Tuple, Optional

# Files to fix - these all have the _initialize_state error
FILES_TO_FIX = {
    "sifaka/adapters/chain/formatter.py": "FormatterAdapter",
    "sifaka/adapters/chain/improver.py": "ImproverAdapter",
    "sifaka/adapters/chain/model.py": "ModelAdapter",
    "sifaka/adapters/chain/validator.py": "ValidatorAdapter",
    "sifaka/chain/engine.py": "Engine",
    "sifaka/chain/managers/cache.py": "CacheManager",
    "sifaka/chain/managers/retry.py": "RetryManager",
    "sifaka/classifiers/engine.py": "Engine",
    "sifaka/rules/managers/validation.py": "ValidationManager",
    "sifaka/rules/validators.py": "BaseValidator",
}


def fix_file(filepath: str, class_name: str) -> bool:
    """
    Fix the file by making the class inherit from InitializeStateMixin.

    Args:
        filepath: Path to the file to fix
        class_name: Name of the class to modify

    Returns:
        True if the file was modified, False otherwise
    """
    try:
        # Read file
        with open(filepath, "r") as f:
            content = f.read()

        # Check if the file already imports InitializeStateMixin
        if (
            "from ...utils.mixins import InitializeStateMixin" in content
            or "from ..utils.mixins import InitializeStateMixin" in content
        ):
            print(f"File already has mixin import: {filepath}")
        else:
            # Add the import
            if "from ...utils" in content:
                # Add to existing import from ...utils
                content = re.sub(
                    r"from \.\.\.utils\.([a-zA-Z0-9_]+) import ([^\\]+)",
                    r"from ...utils.\1 import \2\nfrom ...utils.mixins import InitializeStateMixin",
                    content,
                    count=1,
                )
            elif "from ..utils" in content:
                # Add to existing import from ..utils
                content = re.sub(
                    r"from \.\.utils\.([a-zA-Z0-9_]+) import ([^\\]+)",
                    r"from ..utils.\1 import \2\nfrom ..utils.mixins import InitializeStateMixin",
                    content,
                    count=1,
                )
            else:
                # Find all imports and add after the last one
                import_lines = re.findall(r"^import .*$|^from .*$", content, re.MULTILINE)
                if import_lines:
                    last_import = import_lines[-1]
                    # Determine the appropriate import path based on file location
                    if filepath.count("/") >= 3:
                        # Deep in the directory structure
                        import_line = "from ...utils.mixins import InitializeStateMixin"
                    else:
                        # Near the root
                        import_line = "from ..utils.mixins import InitializeStateMixin"

                    content = content.replace(last_import, f"{last_import}\n{import_line}")

        # Check if the class already inherits from InitializeStateMixin
        class_pattern = rf"class {class_name}\(([^)]*)\)"
        matches = re.search(class_pattern, content)

        if matches:
            current_inheritance = matches.group(1)
            if "InitializeStateMixin" in current_inheritance:
                print(f"Class already inherits from InitializeStateMixin: {filepath}")
            else:
                # Add InitializeStateMixin to inheritance
                if current_inheritance:
                    # There's already inheritance, add to it
                    new_inheritance = f"InitializeStateMixin, {current_inheritance}"
                else:
                    # No inheritance yet
                    new_inheritance = "InitializeStateMixin"

                content = re.sub(class_pattern, f"class {class_name}({new_inheritance})", content)

                # Write the modified content back
                with open(filepath, "w") as f:
                    f.write(content)

                print(f"Fixed: {filepath}")
                return True
        else:
            print(f"Could not find class {class_name} in {filepath}")

        return False

    except Exception as e:
        print(f"Error fixing {filepath}: {str(e)}")
        return False


def main() -> None:
    """Fix all files with _initialize_state issue by adding InitializeStateMixin."""
    # Change to the project root directory if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(script_dir, "sifaka")):
        # We're not in the project root, try to find it
        if os.path.exists("sifaka"):
            # We're already in the project root
            pass
        else:
            print("Script must be run from the project root directory")
            sys.exit(1)

    # Fix all files
    fixed_count = 0
    for filepath, class_name in FILES_TO_FIX.items():
        if fix_file(filepath, class_name):
            fixed_count += 1

    print(f"Fixed {fixed_count} files of {len(FILES_TO_FIX)}")


if __name__ == "__main__":
    main()
