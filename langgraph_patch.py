"""
Temporary patch for langgraph.py to fix the StateGraph import.
"""

import os
import re


def patch_langgraph_module():
    """Patch the langgraph.py module to fix the StateGraph import."""
    # Path to the langgraph.py file
    langgraph_path = os.path.join("sifaka", "integrations", "langgraph.py")

    # Read the file content
    with open(langgraph_path, "r") as f:
        content = f.read()

    # Replace the import statement
    new_content = re.sub(
        r"from langchain\.graphs import StateGraph",
        "from langgraph.graph import StateGraph",
        content,
    )

    # Write the updated content back to the file
    with open(langgraph_path, "w") as f:
        f.write(new_content)

    print(f"Patched {langgraph_path} to use correct StateGraph import")


if __name__ == "__main__":
    patch_langgraph_module()
