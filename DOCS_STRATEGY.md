# Documentation Strategy (v0.2.0)

## Current State

Sifaka has comprehensive documentation in `docs/` directory (24+ files, ~5,500 lines):
- User guides (basic-usage, advanced-usage, critics, validators, configuration)
- Plugin development guides (development, API reference, best practices)
- Architecture documentation
- Development guides (setup, environment, commit instructions)
- FAQ and reference materials

## Modernization Plan

### Phase 1 (v0.2.0) - ✅ Complete
- [x] Created AGENTS.md as primary AI agent entry point
- [x] Added CLAUDE.md and CURSOR.md symlinks
- [x] Updated dependencies to latest versions (PydanticAI 1.14+)
- [x] Maintained existing comprehensive docs

### Phase 2 (v0.3.0) - Future
- [ ] Consolidate overlapping documentation
- [ ] Reduce plugin development docs volume (currently 39K lines)
- [ ] Move critical developer info to AGENTS.md
- [ ] Keep user-facing docs in README.md
- [ ] Archive or remove redundant guides

### Phase 3 (v0.4.0) - Future
- [ ] Evaluate minimal docs approach (like arbiter)
- [ ] Consider removing docs/ directory entirely
- [ ] Inline all documentation in README.md and AGENTS.md
- [ ] Defer comprehensive docs to v1.0+ if needed

## Current Best Practice

**For AI Agents**: Start with AGENTS.md (comprehensive developer guide)
**For Users**: Start with README.md (user-facing guide with examples)
**For Deep Dives**: Refer to docs/ directory (comprehensive reference)

## Notes

Following arbiter's lead, Sifaka is moving toward a minimalist documentation approach:
- **3 core files**: AGENTS.md (dev), README.md (user), CHANGELOG.md (history)
- **Inline examples**: Examples embedded directly in docs rather than separate files
- **Just-in-time docs**: Document when needed, not speculatively

This transition preserves existing comprehensive docs while establishing modern entry points.
