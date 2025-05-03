# Docstring Standardization Guide

This document tracks the progress of docstring standardization across the Sifaka codebase.

## Overview

As part of improving Sifaka's documentation, we're updating all docstrings to follow a consistent format. This document tracks which files have been updated and which still need work.

## Standardization Process

For each file:
1. Review existing docstrings
2. Update to follow [Google Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
3. Ensure all public classes and methods have docstrings
4. Add type annotations where missing
5. Mark as complete in this document

## Progress by Module

Use the following status indicators:
- ✅ Complete - All docstrings follow standards
- 🟡 Partial - Some docstrings follow standards
- 🔴 Not Started - No docstrings follow standards
- 🚫 N/A - File doesn't need docstrings (e.g., empty `__init__.py`)

### Core Modules

- [x] sifaka/__init__.py
- [ ] sifaka/chain/__init__.py
- [ ] sifaka/chain/core.py
- [ ] sifaka/chain/factories.py
- [ ] sifaka/chain/orchestrator.py
- [ ] sifaka/chain/result.py
- [ ] sifaka/validation.py
- [ ] sifaka/generation.py
- [ ] sifaka/improvement.py

### Rules

- [ ] sifaka/rules/__init__.py
- [ ] sifaka/rules/base.py
- [ ] sifaka/rules/content/__init__.py
- [ ] sifaka/rules/content/base.py
- [ ] sifaka/rules/content/prohibited.py
- [ ] sifaka/rules/content/safety.py
- [ ] sifaka/rules/content/sentiment.py
- [ ] sifaka/rules/content/tone.py
- [ ] sifaka/rules/domain/__init__.py
- [ ] sifaka/rules/domain/base.py
- [ ] sifaka/rules/domain/consistency.py
- [ ] sifaka/rules/domain/legal.py
- [ ] sifaka/rules/domain/medical.py
- [ ] sifaka/rules/domain/python.py
- [ ] sifaka/rules/factual/__init__.py
- [ ] sifaka/rules/factual/accuracy.py
- [ ] sifaka/rules/factual/base.py
- [ ] sifaka/rules/factual/citation.py
- [ ] sifaka/rules/factual/confidence.py
- [ ] sifaka/rules/factual/consistency.py
- [ ] sifaka/rules/formatting/__init__.py
- [ ] sifaka/rules/formatting/format.py
- [ ] sifaka/rules/formatting/length.py
- [ ] sifaka/rules/formatting/style.py
- [ ] sifaka/rules/formatting/whitespace.py

### Models

- [ ] sifaka/models/__init__.py
- [ ] sifaka/models/anthropic.py
- [ ] sifaka/models/base.py
- [ ] sifaka/models/core.py
- [ ] sifaka/models/gemini.py
- [ ] sifaka/models/mock.py
- [ ] sifaka/models/openai.py
- [ ] sifaka/models/managers/__init__.py
- [ ] sifaka/models/managers/client.py
- [ ] sifaka/models/managers/token_counter.py
- [ ] sifaka/models/managers/tracing.py
- [ ] sifaka/models/services/__init__.py
- [ ] sifaka/models/services/generation.py

### Classifiers

- [ ] sifaka/classifiers/__init__.py
- [ ] sifaka/classifiers/base.py
- [ ] sifaka/classifiers/bias.py
- [ ] sifaka/classifiers/genre.py
- [ ] sifaka/classifiers/language.py
- [ ] sifaka/classifiers/ner.py
- [ ] sifaka/classifiers/profanity.py
- [ ] sifaka/classifiers/readability.py
- [ ] sifaka/classifiers/sentiment.py
- [ ] sifaka/classifiers/spam.py
- [ ] sifaka/classifiers/topic.py
- [ ] sifaka/classifiers/toxicity.py
- [ ] sifaka/classifiers/toxicity_model.py

### Critics

- [ ] sifaka/critics/__init__.py
- [ ] sifaka/critics/base.py
- [ ] sifaka/critics/core.py
- [ ] sifaka/critics/factories.py
- [ ] sifaka/critics/models.py
- [ ] sifaka/critics/prompt.py
- [ ] sifaka/critics/protocols.py
- [ ] sifaka/critics/reflexion.py
- [ ] sifaka/critics/style.py
- [ ] sifaka/critics/managers/__init__.py
- [ ] sifaka/critics/managers/memory.py
- [ ] sifaka/critics/managers/prompt.py
- [ ] sifaka/critics/managers/prompt_factories.py
- [ ] sifaka/critics/managers/response.py
- [ ] sifaka/critics/services/__init__.py
- [ ] sifaka/critics/services/critique.py

### Adapters

- [ ] sifaka/adapters/__init__.py
- [ ] sifaka/adapters/rules/__init__.py
- [ ] sifaka/adapters/rules/base.py
- [ ] sifaka/adapters/rules/classifier.py
- [ ] sifaka/adapters/rules/guardrails_adapter.py

### Utils

- [ ] sifaka/utils/__init__.py
- [ ] sifaka/utils/logging.py
- [ ] sifaka/utils/tracing.py
- [ ] sifaka/utils/validation.py
- [ ] sifaka/utils/patches/__init__.py

### Chain Components

- [ ] sifaka/chain/formatters/__init__.py
- [ ] sifaka/chain/formatters/result.py
- [ ] sifaka/chain/managers/__init__.py
- [ ] sifaka/chain/managers/prompt.py
- [ ] sifaka/chain/managers/validation.py
- [ ] sifaka/chain/strategies/__init__.py
- [ ] sifaka/chain/strategies/retry.py

## Prioritization

Priority order for standardization:
1. Core modules
2. Models
3. Rules
4. Classifiers
5. Critics
6. Adapters
7. Utils
8. Chain components

## Contribution Guidelines

When contributing docstring standardization:
1. Work on one file at a time
2. Follow the [Docstring Style Guide](./docstring_style_guide.md)
3. Submit a PR with changes to one module
4. Update this document to mark progress
5. Request review from documentation team
