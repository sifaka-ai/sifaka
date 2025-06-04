"""Integration with PydanticAI's common tools.

This module provides easy access to PydanticAI's built-in common tools
with Sifaka-specific configuration and error handling.
"""

import os
from typing import List, Optional, Dict, Any, Literal

from pydantic_ai.tools import Tool
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

# Import PydanticAI common tools with fallback handling
AVAILABLE_TOOLS = {}

try:
    from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
    AVAILABLE_TOOLS["duckduckgo"] = duckduckgo_search_tool
except ImportError:
    logger.debug("DuckDuckGo search tool not available")

try:
    from pydantic_ai.common_tools.tavily import tavily_search_tool
    AVAILABLE_TOOLS["tavily"] = tavily_search_tool
except ImportError:
    logger.debug("Tavily search tool not available")


def get_pydantic_ai_common_tools(
    tools: List[Literal["duckduckgo", "tavily"]] = None,
    duckduckgo_config: Optional[Dict[str, Any]] = None,
    tavily_config: Optional[Dict[str, Any]] = None,
) -> List[Tool]:
    """Get PydanticAI common tools with Sifaka configuration.
    
    Args:
        tools: List of tools to include (default: all available)
        duckduckgo_config: Configuration for DuckDuckGo tool
        tavily_config: Configuration for Tavily tool
        
    Returns:
        List of configured PydanticAI Tool instances
        
    Example:
        ```python
        # Get all available tools
        tools = get_pydantic_ai_common_tools()
        
        # Get specific tools with configuration
        tools = get_pydantic_ai_common_tools(
            tools=["duckduckgo", "tavily"],
            duckduckgo_config={"max_results": 10},
            tavily_config={"api_key": "your-key", "max_results": 5}
        )
        ```
    """
    if tools is None:
        tools = list(AVAILABLE_TOOLS.keys())
    
    result_tools = []
    
    for tool_name in tools:
        if tool_name not in AVAILABLE_TOOLS:
            logger.warning(f"Tool '{tool_name}' not available, skipping")
            continue
        
        tool_func = AVAILABLE_TOOLS[tool_name]
        
        try:
            if tool_name == "duckduckgo":
                config = duckduckgo_config or {}
                tool = tool_func(**config)
                result_tools.append(tool)
                logger.debug("Added DuckDuckGo search tool")
            
            elif tool_name == "tavily":
                config = tavily_config or {}
                
                # Handle API key
                if "api_key" not in config:
                    api_key = os.getenv("TAVILY_API_KEY")
                    if api_key:
                        config["api_key"] = api_key
                    else:
                        logger.warning("Tavily API key not found, skipping Tavily tool")
                        continue
                
                tool = tool_func(**config)
                result_tools.append(tool)
                logger.debug("Added Tavily search tool")
        
        except Exception as e:
            logger.error(f"Failed to create {tool_name} tool: {e}")
    
    return result_tools


def list_available_common_tools() -> List[str]:
    """List all available PydanticAI common tools.
    
    Returns:
        List of available tool names
    """
    return list(AVAILABLE_TOOLS.keys())


def is_tool_available(tool_name: str) -> bool:
    """Check if a specific tool is available.
    
    Args:
        tool_name: Name of the tool to check
        
    Returns:
        True if tool is available, False otherwise
    """
    return tool_name in AVAILABLE_TOOLS


def get_tool_requirements(tool_name: str) -> Dict[str, Any]:
    """Get requirements for a specific tool.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Dictionary with tool requirements
    """
    requirements = {
        "duckduckgo": {
            "package": "pydantic-ai-slim[duckduckgo]",
            "requires_auth": False,
            "env_vars": [],
        },
        "tavily": {
            "package": "pydantic-ai-slim[tavily]",
            "requires_auth": True,
            "env_vars": ["TAVILY_API_KEY"],
        },
    }
    
    return requirements.get(tool_name, {})


def validate_tool_setup(tool_name: str) -> Dict[str, Any]:
    """Validate that a tool is properly set up.
    
    Args:
        tool_name: Name of the tool to validate
        
    Returns:
        Dictionary with validation results
    """
    result = {
        "tool_name": tool_name,
        "available": False,
        "properly_configured": False,
        "missing_requirements": [],
        "warnings": [],
    }
    
    # Check if tool is available
    if not is_tool_available(tool_name):
        requirements = get_tool_requirements(tool_name)
        result["missing_requirements"].append(f"Install: pip install '{requirements.get('package', 'unknown')}'")
        return result
    
    result["available"] = True
    
    # Check tool-specific requirements
    if tool_name == "tavily":
        if not os.getenv("TAVILY_API_KEY"):
            result["missing_requirements"].append("Set TAVILY_API_KEY environment variable")
        else:
            result["properly_configured"] = True
    else:
        result["properly_configured"] = True
    
    return result


def get_setup_instructions() -> Dict[str, Dict[str, Any]]:
    """Get setup instructions for all common tools.
    
    Returns:
        Dictionary with setup instructions for each tool
    """
    instructions = {}
    
    for tool_name in ["duckduckgo", "tavily"]:
        validation = validate_tool_setup(tool_name)
        requirements = get_tool_requirements(tool_name)
        
        instructions[tool_name] = {
            "validation": validation,
            "requirements": requirements,
            "setup_commands": [],
        }
        
        if not validation["available"]:
            package = requirements.get("package", "unknown")
            instructions[tool_name]["setup_commands"].append(f"pip install '{package}'")
        
        if validation["missing_requirements"]:
            instructions[tool_name]["setup_commands"].extend(validation["missing_requirements"])
    
    return instructions
