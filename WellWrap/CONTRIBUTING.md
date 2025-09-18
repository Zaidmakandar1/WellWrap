# Contributing to WellWrap

Thank you for your interest in contributing to WellWrap! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues
- Use GitHub Issues to report bugs or request features
- Provide detailed information about the issue
- Include steps to reproduce the problem
- Add relevant screenshots or error messages

### Submitting Changes
1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes** with clear, descriptive commits
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Submit a pull request**

## 🛠️ Development Setup

### Prerequisites
- Python 3.8+
- Git
- Virtual environment (recommended)

### Local Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/wellwrap.git
cd wellwrap

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
python init_database.py --init

# Run tests
python -m pytest tests/
```

## 📝 Code Style Guidelines

### Python Code
- Follow PEP 8 style guide
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Maximum line length: 88 characters (Black formatter)

### Example:
```python
def analyze_medical_report(report_text: str) -> Dict[str, Any]:
    """
    Analyze medical report text and extract health insights.
    
    Args:
        report_text: Raw text from medical report
        
    Returns:
        Dictionary containing analysis results and health score
    """
    # Implementation here
    pass
```

### HTML/CSS
- Use semantic HTML elements
- Follow Bootstrap conventions
- Keep CSS organized and commented
- Use consistent naming conventions

### JavaScript
- Use modern ES6+ syntax
- Add comments for complex logic
- Follow consistent formatting
- Handle errors gracefully

## 🧪 Testing Guidelines

### Writing Tests
- Write tests for all new functionality
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies

### Test Structure
```python
def test_medical_analysis_with_valid_data():
    """Test medical analysis with valid input data."""
    # Arrange
    sample_text = "Hemoglobin: 12.5 g/dL"
    
    # Act
    result = analyze_medical_report(sample_text)
    
    # Assert
    assert result['health_score'] > 0
    assert len(result['test_results']) > 0
```

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_medical_analysis.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 🏗️ Project Structure

### Adding New Features
- Place backend logic in `backend/`
- Add templates to `frontend/templates/`
- Put ML models in `ml/models/`
- Add tests to `tests/`
- Update documentation in `docs/`

### File Organization
```
new-feature/
├── backend/services/new_feature_service.py
├── frontend/templates/new_feature/
├── tests/test_new_feature.py
└── docs/new_feature.md
```

## 🔒 Security Guidelines

### Data Handling
- Never commit sensitive data (passwords, API keys)
- Use environment variables for configuration
- Validate all user inputs
- Sanitize data before database operations

### Medical Data
- Follow HIPAA compliance guidelines
- Implement proper access controls
- Use encryption for sensitive data
- Log access to medical information

## 📚 Documentation

### Code Documentation
- Add docstrings to all public functions
- Include type hints where possible
- Document complex algorithms
- Provide usage examples

### User Documentation
- Update README.md for new features
- Add API documentation for new endpoints
- Create user guides for complex features
- Include screenshots for UI changes

## 🚀 Pull Request Process

### Before Submitting
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated
- [ ] No sensitive data in commits
- [ ] Branch is up to date with main

### PR Description Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Tests added/updated
- [ ] Manual testing completed
- [ ] No breaking changes

## Screenshots (if applicable)
Add screenshots for UI changes
```

## 🏷️ Commit Message Guidelines

### Format
```
type(scope): brief description

Detailed explanation if needed

Fixes #issue-number
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

### Examples
```
feat(analysis): add diabetes risk detection algorithm

Implement new algorithm to detect diabetes risk based on glucose
levels and other metabolic indicators.

Fixes #123
```

## 🎯 Areas for Contribution

### High Priority
- [ ] Enhanced medical analysis algorithms
- [ ] Improved error handling
- [ ] Performance optimizations
- [ ] Mobile responsiveness
- [ ] Accessibility improvements

### Medium Priority
- [ ] Additional test coverage
- [ ] API documentation
- [ ] Internationalization
- [ ] Advanced reporting features
- [ ] Integration with external APIs

### Good First Issues
- [ ] UI/UX improvements
- [ ] Documentation updates
- [ ] Bug fixes
- [ ] Code cleanup
- [ ] Adding tests

## 🤔 Questions?

If you have questions about contributing:
- Check existing GitHub Issues and Discussions
- Review the documentation in `/docs`
- Ask questions in GitHub Discussions
- Contact maintainers through GitHub

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- GitHub contributors page

Thank you for helping make healthcare more accessible! 🏥❤️