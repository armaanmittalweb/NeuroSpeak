# Contributing to NeuroSpeak

Thank you for your interest in contributing to NeuroSpeak! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How Can I Contribute?

### Reporting Bugs

Bug reports help us improve NeuroSpeak. When creating a bug report, please include:

1. A clear and descriptive title
2. Steps to reproduce the issue
3. Expected behavior vs. actual behavior
4. System information (OS, Python version, etc.)
5. Screenshots or logs if applicable

### Suggesting Enhancements

We welcome ideas for enhancements! When suggesting features:

1. Provide a clear description of the feature
2. Explain why it would be useful
3. Include examples of how it would work
4. Mention similar features in other projects if relevant

### Pull Requests

1. Fork the repository
2. Create a new branch for your changes
3. Make your changes following our coding standards
4. Write tests for your changes when applicable
5. Ensure all tests pass
6. Submit a pull request

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/yourusername/NeuroSpeak.git
cd NeuroSpeak
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

4. Run tests to verify your setup:
```bash
pytest
```

## Coding Standards

- Follow PEP 8 style guidelines
- Write docstrings for all functions, classes, and modules
- Include type hints where appropriate
- Write unit tests for new functionality
- Keep lines under 100 characters where possible

## Testing

- Add tests for new features or bug fixes
- Ensure all tests pass before submitting pull requests
- Run tests using pytest:
```bash
pytest
```

## Documentation

- Update documentation when changing functionality
- Follow Google-style docstrings
- Include examples where helpful

## Commit Messages

- Use clear and descriptive commit messages
- Begin with a capitalized verb in imperative mood (e.g., "Add", "Fix", "Update")
- Reference issue numbers when applicable

## Licensing

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

## Questions?

If you have any questions, please open an issue or contact the project maintainers.

Thank you for contributing to NeuroSpeak!