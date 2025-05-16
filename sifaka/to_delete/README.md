# Files to Delete

The following files have been refactored and moved to the central interfaces directory. After verifying that all imports have been updated correctly, these files can be safely deleted.

## Chain Interfaces
- `sifaka/chain/interfaces.py` - Moved to `sifaka/interfaces/chain_components.py`

## Classifier Interfaces
- `sifaka/classifiers/interfaces.py` - Moved to `sifaka/interfaces/classifier.py`

Ensure that all imports in the codebase have been updated to use the new centralized interfaces. Run tests to verify that the refactoring hasn't introduced any regressions before deleting these files.

This refactoring helps eliminate circular dependencies by centralizing all interface definitions in a single location, ensuring that interface-only files are imported without bringing in any of their implementations.