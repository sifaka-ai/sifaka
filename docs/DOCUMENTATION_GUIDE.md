# Documentation Maintenance Guide

## Overview

Sifaka uses MkDocs with the Material theme for documentation. All documentation lives in the `docs/` folder and is built/served using MkDocs.

## Structure

```
sifaka/
├── mkdocs.yml          # MkDocs configuration
├── docs/               # All documentation source files
│   ├── index.md        # Homepage
│   ├── quickstart.md   # Getting started guide
│   ├── guide/          # User guides
│   ├── critics/        # Critic-specific docs
│   ├── api/            # API reference (auto-generated)
│   └── dev/            # Developer documentation
└── site/               # Built documentation (git-ignored)
```

## Setup

1. **Install MkDocs and dependencies:**
   ```bash
   uv pip install mkdocs mkdocs-material mkdocstrings[python]
   ```

2. **Serve documentation locally:**
   ```bash
   mkdocs serve
   ```
   Visit http://127.0.0.1:8000 to see live preview

3. **Build static site:**
   ```bash
   mkdocs build
   ```

## Maintaining Documentation

### When to Update Docs

- **After adding new features**: Document in appropriate guide
- **After API changes**: Update API reference
- **After fixing bugs**: Update troubleshooting/FAQ if relevant
- **After changing behavior**: Update relevant guides

### How to Update

1. **For content changes:**
   - Edit the relevant `.md` file in `docs/`
   - Run `mkdocs serve` to preview changes
   - Commit changes with descriptive message

2. **For API documentation:**
   - API docs are auto-generated from docstrings
   - Update docstrings in Python code
   - MkDocs will pick up changes automatically

3. **For new pages:**
   - Create new `.md` file in appropriate directory
   - Add to `nav` section in `mkdocs.yml`
   - Link from other relevant pages

### Documentation Standards

1. **Use clear headings:**
   ```markdown
   # Page Title
   ## Major Section
   ### Subsection
   ```

2. **Include code examples:**
   ```markdown
   ```python
   from sifaka import improve

   result = await improve("Your text here")
   ```
   ```

3. **Add warnings/notes:**
   ```markdown
   !!! warning "Important"
       This feature requires Python 3.9+

   !!! note
       See the [configuration guide](../guide/configuration.md) for details.
   ```

4. **Keep it updated:**
   - Review docs when changing code
   - Test all code examples
   - Update version numbers

### Auto-Generated API Docs

Create files like `docs/api/core.md`:

```markdown
# Core API

::: sifaka.api
    options:
      show_source: true
      show_bases: true
```

This will automatically document the `sifaka.api` module.

### Deployment

**GitHub Pages (recommended):**

1. **Add GitHub Action** (`.github/workflows/docs.yml`):
   ```yaml
   name: Deploy Docs
   on:
     push:
       branches: [main]

   jobs:
     deploy:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         - uses: actions/setup-python@v4
           with:
             python-version: '3.x'
         - run: pip install mkdocs-material mkdocstrings[python]
         - run: mkdocs gh-deploy --force
   ```

2. **Enable GitHub Pages** in repo settings
3. Docs will auto-deploy on push to main

### Tips for Good Documentation

1. **Think like a user**: What would you want to know?
2. **Show, don't just tell**: Include examples
3. **Be concise**: Get to the point quickly
4. **Cross-reference**: Link between related topics
5. **Test everything**: Ensure code examples work

### Common Tasks

**Add a new critic guide:**
```bash
# Create the file
echo "# New Critic Guide" > docs/critics/new-critic.md

# Add to mkdocs.yml nav section
# Update selection guide to mention it
```

**Update changelog:**
```bash
# Add entry to docs/about/changelog.md
# Follow semantic versioning
# Include breaking changes prominently
```

**Fix broken links:**
```bash
# MkDocs will warn about broken links
mkdocs build --strict
```

## Questions?

- Check MkDocs documentation: https://www.mkdocs.org/
- Material theme docs: https://squidfunk.github.io/mkdocs-material/
- Ask in project discussions
