# Contributing to Sifaka

Thank you for your interest in contributing to Sifaka! This document provides guidelines and instructions for contributing to the project.

## Development Workflow

### 1. Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the upstream remote

```bash
git clone https://github.com/yourusername/sifaka.git
cd sifaka
git remote add upstream https://github.com/original/sifaka.git
```

### 2. Create a Branch

Create a feature branch for your changes:

```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes

Follow these guidelines when making changes:

- Write clear, concise commit messages
- Follow the code style guide
- Add tests for new features
- Update documentation
- Keep changes focused and small

### 4. Run Tests

```bash
# Run all tests
python -m pytest

# Run specific test category
python -m pytest tests/edge_cases/
python -m pytest tests/performance/
python -m pytest tests/security/
python -m pytest tests/integration/
```

### 5. Submit Pull Request

1. Push your changes to your fork
2. Create a pull request
3. Fill out the pull request template
4. Request review from maintainers

## Code Style

### Python Style Guide

Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with these additions:

- Use type hints
- Document all public functions and classes
- Keep functions small and focused
- Use meaningful variable names

### Documentation Style

- Use Google-style docstrings
- Include examples in docstrings
- Keep documentation up to date
- Use clear, concise language

## Testing Guidelines

### Test Categories

1. **Edge Cases**
   - Test boundary conditions
   - Test error handling
   - Test recovery scenarios

2. **Performance**
   - Test response times
   - Test resource usage
   - Test scalability

3. **Security**
   - Test input validation
   - Test output sanitization
   - Test authentication

4. **Integration**
   - Test component interactions
   - Test full workflows
   - Test error propagation

### Test Requirements

- All new features must have tests
- Tests must be deterministic
- Tests must be fast
- Tests must be isolated
- Tests must be documented

## Documentation Requirements

### Required Documentation

1. **Code Documentation**
   - Function and class docstrings
   - Type hints
   - Examples

2. **User Documentation**
   - Setup instructions
   - Usage examples
   - Configuration guide

3. **Developer Documentation**
   - Architecture overview
   - API reference
   - Contributing guide

## Review Process

### Pull Request Review

1. **Initial Review**
   - Code style
   - Test coverage
   - Documentation
   - Functionality

2. **Technical Review**
   - Architecture
   - Performance
   - Security
   - Scalability

3. **Final Review**
   - Integration
   - Deployment
   - Documentation
   - Release notes

### Review Guidelines

- Be constructive
- Be specific
- Be timely
- Be thorough

## Release Process

### Versioning

Follow [Semantic Versioning](https://semver.org/):
- MAJOR: Incompatible API changes
- MINOR: Backwards-compatible functionality
- PATCH: Backwards-compatible bug fixes

### Release Steps

1. Update version number
2. Update changelog
3. Run all tests
4. Create release branch
5. Tag release
6. Build and publish
7. Update documentation

## Community Guidelines

### Code of Conduct

- Be respectful
- Be inclusive
- Be constructive
- Be professional

### Communication

- Use clear language
- Be responsive
- Be helpful
- Be patient

## Getting Help

- Check the documentation
- Open an issue
- Join the community chat
- Contact maintainers

## Recognition

Contributors will be:
- Listed in the contributors file
- Acknowledged in release notes
- Invited to join the team