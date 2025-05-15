# Async Removal Plan

## Overview
This plan outlines the process for removing async functionality from the Sifaka codebase. The async code was never fully implemented, so this removal process should be relatively straightforward.

## Current Status
✅ Phase 1: Classifier System - COMPLETED
- Removed async-related code from classifier configuration
- Updated documentation to remove async references
- Removed async-related methods from interfaces
- Removed async-related code from engine
- Removed async-related code from adapters
- Removed async-related code from base classes
- Removed async-related code from factories
- Updated README.md to remove async references

✅ Phase 2: Chain System - COMPLETED
- Chain Components
  - [x] Removed async from model component
  - [x] Removed async from validator component
  - [x] Removed async from formatter component
  - [x] Removed async from improver component
  - [x] Removed async from async_chain
- Chain Adapters
  - [x] Removed async from model adapter
  - [x] Removed async from improver adapter
  - [x] Removed async from validator adapter
  - [x] Removed async from formatter adapter
- Documentation
  - [x] Updated chain README.md
  - [x] Updated component documentation
  - [x] Updated adapter documentation

⏳ Phase 3: Critics System - PENDING
- Core Services
  - [ ] Remove async from critique service
  - [ ] Update service documentation
- Implementations
  - [ ] Remove async from self_refine implementation
  - [ ] Remove async from reflexion implementation
  - [ ] Remove async from constitutional implementation
  - [ ] Remove async from prompt implementation
  - [ ] Remove async from self_rag implementation
  - [ ] Remove async from lac implementation
- Documentation
  - [ ] Update critics README.md
  - [ ] Update implementation documentation

⏳ Phase 4: Retrieval System - PENDING
- Core Components
  - [ ] Remove async from retrieval interface
  - [ ] Update interface documentation
- Documentation
  - [ ] Update retrieval README.md
  - [ ] Update interface documentation

⏳ Phase 5: Model System - PENDING
- Core Interfaces
  - [ ] Remove async from model interface
  - [ ] Remove async from rule interface
  - [ ] Remove async from critic interface
- Documentation
  - [ ] Update model README.md
  - [ ] Update interface documentation

## Implementation Details

### Phase 1: Classifier System ✅
1. Classifier Configuration
   - Removed async-related configuration options
   - Updated documentation to remove async references
   - Simplified configuration structure

2. Classifier Interfaces
   - Removed async-related methods
   - Updated documentation to remove async references
   - Simplified interface structure

3. Classifier Engine
   - Removed async-related methods
   - Updated documentation to remove async references
   - Simplified engine structure

4. Classifier Adapters
   - Removed async-related methods
   - Updated documentation to remove async references
   - Simplified adapter structure

5. Base Classes
   - Removed async-related methods
   - Updated documentation to remove async references
   - Simplified base class structure

6. Factory Functions
   - Removed async-related parameters
   - Updated documentation to remove async references
   - Simplified factory structure

7. README.md
   - Removed async-related examples
   - Updated to reflect current synchronous-only implementation
   - Simplified documentation structure

### Phase 2: Chain System ✅
1. Chain Components
   - Removed async method declarations
   - Removed async-related configuration
   - Updated component documentation
   - Simplified component structure

2. Chain Adapters
   - Removed async method declarations
   - Removed async-related configuration
   - Updated adapter documentation
   - Simplified adapter structure

3. Documentation
   - Updated chain README.md
   - Updated component documentation
   - Updated adapter documentation
   - Removed async examples

### Phase 3: Critics System ⏳
1. Core Services
   - Remove async method declarations
   - Remove async-related configuration
   - Update service documentation
   - Simplify service structure

2. Implementations
   - Remove async method declarations
   - Remove async-related configuration
   - Update implementation documentation
   - Simplify implementation structure

3. Documentation
   - Update critics README.md
   - Update implementation documentation
   - Remove async examples

### Phase 4: Retrieval System ⏳
1. Core Components
   - Remove async method declarations
   - Remove async-related configuration
   - Update interface documentation
   - Simplify interface structure

2. Documentation
   - Update retrieval README.md
   - Update interface documentation
   - Remove async examples

### Phase 5: Model System ⏳
1. Core Interfaces
   - Remove async method declarations
   - Remove async-related configuration
   - Update interface documentation
   - Simplify interface structure

2. Documentation
   - Update model README.md
   - Update interface documentation
   - Remove async examples

## Success Criteria
- [x] All async-related code has been removed from classifier system
- [x] All async-related code has been removed from chain system
- [ ] All async-related code has been removed from critics system
- [ ] All async-related code has been removed from retrieval system
- [ ] All async-related code has been removed from model system
- [ ] All tests pass
- [ ] Documentation is up to date
- [ ] No regression in functionality
- [ ] Code is cleaner and simpler

## Timeline
- Phase 1: Classifier System - COMPLETED
- Phase 2: Chain System - COMPLETED
- Phase 3: Critics System - PENDING
- Phase 4: Retrieval System - PENDING
- Phase 5: Model System - PENDING

## Next Steps
1. Begin removing async code from critics system
2. Update critics documentation
3. Move on to retrieval system
4. Continue with model system
5. Final review and cleanup

## Dependencies
The removal order has been chosen based on the following dependencies:
1. Chain System depends on Model System
2. Critics System depends on Model System
3. Retrieval System is independent
4. Model System is independent

This order ensures that we can remove async code without breaking dependencies between systems.