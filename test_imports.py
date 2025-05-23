#!/usr/bin/env python3
"""
Test script to debug import issues.
"""

import sys
import os

# Add the sifaka directory to the path
sys.path.insert(0, '/Users/evanvolgas/Documents/not_beam/sifaka')

def test_basic_imports():
    """Test basic Python imports."""
    print("Testing basic imports...")
    
    try:
        import datetime
        print("✅ datetime")
    except Exception as e:
        print(f"❌ datetime: {e}")
        return False
    
    try:
        import uuid
        print("✅ uuid")
    except Exception as e:
        print(f"❌ uuid: {e}")
        return False
    
    try:
        from typing import Any, Dict, List, Optional
        print("✅ typing")
    except Exception as e:
        print(f"❌ typing: {e}")
        return False
    
    try:
        from pydantic import BaseModel, Field
        print("✅ pydantic")
    except Exception as e:
        print(f"❌ pydantic: {e}")
        return False
    
    return True

def test_sifaka_imports():
    """Test sifaka imports step by step."""
    print("\nTesting sifaka imports...")
    
    try:
        from sifaka.core.thought import Document
        print("✅ Document")
    except Exception as e:
        print(f"❌ Document: {e}")
        return False
    
    try:
        from sifaka.core.thought import ValidationResult
        print("✅ ValidationResult")
    except Exception as e:
        print(f"❌ ValidationResult: {e}")
        return False
    
    try:
        from sifaka.core.thought import CriticFeedback
        print("✅ CriticFeedback")
    except Exception as e:
        print(f"❌ CriticFeedback: {e}")
        return False
    
    try:
        from sifaka.core.thought import Thought
        print("✅ Thought")
    except Exception as e:
        print(f"❌ Thought: {e}")
        return False
    
    try:
        from sifaka.core.interfaces import Model, Validator, Critic, Retriever
        print("✅ Interfaces")
    except Exception as e:
        print(f"❌ Interfaces: {e}")
        return False
    
    try:
        from sifaka.utils.error_handling import ChainError, chain_context
        print("✅ Error handling")
    except Exception as e:
        print(f"❌ Error handling: {e}")
        return False
    
    try:
        from sifaka.utils.logging import get_logger
        print("✅ Logging")
    except Exception as e:
        print(f"❌ Logging: {e}")
        return False
    
    try:
        from sifaka.chain import Chain
        print("✅ Chain")
    except Exception as e:
        print(f"❌ Chain: {e}")
        return False
    
    return True

def main():
    """Run the import tests."""
    print("🔍 Import Debugging Script")
    print("=" * 40)
    
    if not test_basic_imports():
        print("\n❌ Basic imports failed!")
        return
    
    if not test_sifaka_imports():
        print("\n❌ Sifaka imports failed!")
        return
    
    print("\n🎉 All imports successful!")

if __name__ == "__main__":
    main()
