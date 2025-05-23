#!/usr/bin/env python3
"""
Smoke tests for Sifaka persistence.

Quick tests to verify basic functionality is working.
"""

import tempfile
from sifaka.core.thought import Thought
from sifaka.persistence.json import JSONThoughtStorage


def test_basic_smoke():
    """Basic smoke test - can we save and retrieve a thought?"""
    print("Running smoke test...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create storage
        storage = JSONThoughtStorage(storage_dir=temp_dir)
        print("✓ Storage created")
        
        # Create thought
        thought = Thought(prompt="Test prompt", text="Test text")
        print(f"✓ Thought created: {thought.id}")
        
        # Save thought
        storage.save_thought(thought)
        print("✓ Thought saved")
        
        # Retrieve thought
        retrieved = storage.get_thought(thought.id)
        assert retrieved is not None, "Failed to retrieve thought"
        assert retrieved.prompt == "Test prompt", "Prompt mismatch"
        print("✓ Thought retrieved successfully")
        
        # Health check
        health = storage.health_check()
        assert health["status"] == "healthy", "Storage not healthy"
        print("✓ Health check passed")


def test_import_smoke():
    """Test that all required modules can be imported."""
    print("\nTesting imports...")
    
    try:
        from sifaka.core.thought import Thought, Document, ValidationResult, CriticFeedback
        print("✓ Core thought classes imported")
        
        from sifaka.persistence.json import JSONThoughtStorage
        print("✓ JSON storage imported")
        
        from sifaka.persistence.base import ThoughtStorage, ThoughtQuery, PersistenceError
        print("✓ Base persistence classes imported")
        
        from sifaka.persistence import JSONThoughtStorage as ImportedStorage
        print("✓ Package-level imports work")
        
    except ImportError as e:
        raise AssertionError(f"Import failed: {e}")


def main():
    """Run smoke tests."""
    print("🚀 Sifaka Persistence Smoke Tests")
    print("=" * 40)
    
    try:
        test_import_smoke()
        test_basic_smoke()
        
        print("\n" + "=" * 40)
        print("🎉 All smoke tests passed!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
