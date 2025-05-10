1. ✅ **State Management Standardization** (switch from `_state` to `_state_manager`):
   - ✅ Update `sifaka/core/managers/memory.py` (many instances)
   - ✅ Update `sifaka/critics/implementations/lac.py`
   - ✅ Update `sifaka/critics/implementations/reflexion.py`
   - ✅ Update `sifaka/classifiers/implementations/content/sentiment.py`
   - ✅ Update `sifaka/classifiers/implementations/content/spam.py`
   - ✅ Update `sifaka/classifiers/implementations/content/bias.py`
   - ✅ Update `sifaka/classifiers/implementations/content/toxicity.py`
   - ✅ Update `sifaka/classifiers/implementations/content/profanity.py`
   - ✅ Update `sifaka/classifiers/implementations/properties/readability.py`
   - ✅ Update `sifaka/classifiers/implementations/properties/topic.py`
   - ✅ Update `sifaka/classifiers/implementations/properties/genre.py`
   - ✅ Update `sifaka/classifiers/implementations/entities/ner.py`
   - ✅ No changes needed for `sifaka/classifiers/state.py` (already uses StateTracker)
   - ✅ No changes needed for `sifaka/chain/state.py` (already uses StateTracker)
   - ✅ No changes needed for `sifaka/utils/state.py` (defines StateManager)



2. ✅ **Interface Directory Cleanup**: (follow what you did for Chain and Classifiers; do not fuck up state; clean up the old files when you're done; if examples fail, come back to them later. Do not maintain backwards compatibility).
   - ✅ Consolidate `/sifaka/critics/interfaces` into main interfaces directory (already done)
   - ✅ Consolidate `/sifaka/models/interfaces` into main interfaces directory (already done)
   - ✅ Consolidate `/sifaka/retrieval/interfaces` into main interfaces directory
   - ✅ Consolidate `/sifaka/rules/interfaces` into main interfaces directory

✅ State management standardization is now complete. All files have been updated to use `_state_manager` instead of `_state`.

✅ Interface directory cleanup is now complete. All component-specific interfaces have been consolidated into the main interfaces directory.


3. ⬜ **Testing**:
   - ⬜ Add unit tests for all components
   - ⬜ Add integration tests for component interactions
   - ⬜ Add validation tests for configuration
   - ⬜ Add error handling tests

4. ⬜ **Documentation Updates**:
   - ⬜ Add comprehensive docstrings explaining component relationships
   - ⬜ Document interaction patterns between components
   - ⬜ Add architecture diagrams in docstrings
   - ⬜ Clarify dependency relationships


5. ⬜ **Fix v2 References**:
   - ⬜ Update remaining references to v2 in documentation and code examples
   - ⬜ Update README files to reflect new architecture
