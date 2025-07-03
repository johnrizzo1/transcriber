# Contributing Guidelines

Welcome to the AI Voice Agent project! This guide will help you contribute effectively to the project.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Code Standards](#code-standards)
4. [Testing Requirements](#testing-requirements)
5. [Pull Request Process](#pull-request-process)
6. [Issue Guidelines](#issue-guidelines)
7. [Documentation Standards](#documentation-standards)
8. [Release Process](#release-process)
9. [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Poetry for dependency management
- Git for version control
- Basic understanding of async/await patterns
- Familiarity with voice processing concepts (helpful but not required)

### First-Time Setup

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/transcriber.git
   cd transcriber
   ```

2. **Set up development environment**
   ```bash
   # Install Poetry if you haven't already
   curl -sSL https://install.python-poetry.org | python3 -

   # Install dependencies
   poetry install --with dev,test

   # Activate virtual environment
   poetry shell
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

4. **Verify setup**
   ```bash
   # Run tests to ensure everything works
   pytest

   # Run linting
   flake8 transcriber/
   black --check transcriber/
   mypy transcriber/
   ```

### Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following our standards
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run all tests
   pytest

   # Run specific test categories
   pytest tests/unit/
   pytest tests/integration/
   pytest tests/performance/

   # Check code coverage
   pytest --cov=transcriber --cov-report=html
   ```

4. **Commit and push**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   git push origin feature/your-feature-name
   ```

5. **Create pull request**
   - Use our PR template
   - Link related issues
   - Request review from maintainers

## Development Setup

### Environment Configuration

Create a `.env` file for local development:

```bash
# .env
TRANSCRIBER_LOG_LEVEL=DEBUG
TRANSCRIBER_AUDIO_DEVICE=default
TRANSCRIBER_MODEL=llama3.2:3b
TRANSCRIBER_PERFORMANCE_MONITORING=true
```

### Development Dependencies

The project uses several development tools:

```toml
# pyproject.toml [tool.poetry.group.dev.dependencies]
black = "^23.0.0"           # Code formatting
flake8 = "^6.0.0"          # Linting
mypy = "^1.0.0"            # Type checking
pre-commit = "^3.0.0"      # Git hooks
pytest = "^7.0.0"          # Testing framework
pytest-cov = "^4.0.0"      # Coverage reporting
pytest-asyncio = "^0.21.0" # Async testing
```

### IDE Configuration

#### VS Code Settings

```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/.mypy_cache": true
    }
}
```

#### PyCharm Configuration

1. Set Python interpreter to Poetry virtual environment
2. Enable Black as code formatter
3. Configure pytest as test runner
4. Enable mypy type checking

## Code Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Imports**: Use absolute imports, group by standard/third-party/local
- **Docstrings**: Google style for all public functions and classes
- **Type hints**: Required for all public APIs

### Code Formatting

```bash
# Format code with Black
black transcriber/ tests/

# Sort imports with isort
isort transcriber/ tests/

# Check formatting
black --check transcriber/
isort --check-only transcriber/
```

### Linting Rules

```bash
# Run flake8 linting
flake8 transcriber/

# Configuration in setup.cfg
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = .git,__pycache__,.venv
```

### Type Checking

```bash
# Run mypy type checking
mypy transcriber/

# Configuration in pyproject.toml
[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

### Documentation Standards

#### Docstring Format

```python
def process_audio(audio_data: bytes, sample_rate: int = 16000) -> str:
    """Process audio data and return transcription.
    
    Args:
        audio_data: Raw audio bytes in PCM format
        sample_rate: Audio sample rate in Hz
        
    Returns:
        Transcribed text from the audio
        
    Raises:
        AudioProcessingError: If audio processing fails
        ValidationError: If audio data is invalid
        
    Example:
        >>> audio = capture_audio()
        >>> text = process_audio(audio, 16000)
        >>> print(text)
        "Hello, world!"
    """
    # Implementation here
```

#### Class Documentation

```python
class VoiceAgent:
    """Main voice agent for processing audio and text interactions.
    
    The VoiceAgent coordinates between audio processing, language models,
    and tool execution to provide a complete voice assistant experience.
    
    Attributes:
        settings: Configuration settings for the agent
        llm_service: Language model service instance
        tool_registry: Registry of available tools
        
    Example:
        >>> agent = VoiceAgent(settings)
        >>> await agent.initialize()
        >>> response = await agent.process_text("What's 2 + 2?")
        >>> print(response)
        "2 + 2 equals 4"
    """
```

### Error Handling Standards

```python
# Custom exception hierarchy
class TranscriberError(Exception):
    """Base exception for all transcriber errors."""
    pass

class AudioProcessingError(TranscriberError):
    """Raised when audio processing fails."""
    pass

class ToolExecutionError(TranscriberError):
    """Raised when tool execution fails."""
    pass

# Error handling pattern
async def process_request(request: str) -> str:
    """Process a user request with proper error handling."""
    try:
        result = await self._process_internal(request)
        return result
    except AudioProcessingError as e:
        logger.error(f"Audio processing failed: {e}")
        raise
    except ToolExecutionError as e:
        logger.warning(f"Tool execution failed: {e}")
        return "I encountered an error while processing your request."
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise TranscriberError(f"Processing failed: {e}") from e
```

## Testing Requirements

### Test Structure

```
tests/
├── unit/                 # Unit tests for individual components
│   ├── test_audio_vad.py
│   ├── test_config.py
│   └── test_tools_*.py
├── integration/          # Integration tests for component interaction
│   ├── test_voice_pipeline.py
│   └── test_session_management.py
├── performance/          # Performance and benchmark tests
│   ├── test_benchmarks.py
│   └── test_stress.py
└── conftest.py          # Shared test configuration
```

### Test Categories

#### Unit Tests

```python
import pytest
from unittest.mock import Mock, patch
from transcriber.audio.vad import VADProcessor

class TestVADProcessor:
    def setup_method(self):
        """Set up test fixtures."""
        self.vad = VADProcessor()
    
    def test_initialization(self):
        """Test VAD processor initialization."""
        assert self.vad.threshold > 0
        assert self.vad.frame_duration > 0
    
    @patch('transcriber.audio.vad.webrtcvad')
    def test_process_chunk_with_speech(self, mock_vad):
        """Test processing audio chunk containing speech."""
        mock_vad.Vad().is_speech.return_value = True
        
        audio_chunk = b'\x00' * 320  # Mock audio data
        result = self.vad.process_chunk(audio_chunk)
        
        assert result is True
        mock_vad.Vad().is_speech.assert_called_once()
    
    def test_invalid_audio_data(self):
        """Test handling of invalid audio data."""
        with pytest.raises(ValueError):
            self.vad.process_chunk(b'invalid')
```

#### Integration Tests

```python
import pytest
from transcriber.agent.core import VoiceAgent
from transcriber.config import Settings

class TestVoiceAgentIntegration:
    @pytest.fixture
    async def agent(self):
        """Create and initialize voice agent for testing."""
        settings = Settings()
        agent = VoiceAgent(settings)
        await agent.initialize()
        yield agent
        await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_text_processing_flow(self, agent):
        """Test complete text processing flow."""
        response = await agent.process_text("What's 2 + 2?")
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "4" in response
    
    @pytest.mark.asyncio
    async def test_tool_execution_flow(self, agent):
        """Test tool execution through agent."""
        response = await agent.process_text("List files in current directory")
        
        assert isinstance(response, str)
        assert "file" in response.lower() or "directory" in response.lower()
```

#### Performance Tests

```python
import pytest
import time
import asyncio
from transcriber.performance.benchmarks import BenchmarkRunner

class TestPerformance:
    @pytest.mark.performance
    async def test_response_latency(self):
        """Test response latency is within acceptable limits."""
        agent = VoiceAgent(settings)
        await agent.initialize()
        
        try:
            start_time = time.time()
            await agent.process_text("Hello")
            end_time = time.time()
            
            latency = end_time - start_time
            assert latency < 2.0  # Should respond within 2 seconds
        finally:
            await agent.cleanup()
    
    @pytest.mark.performance
    async def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        agent = VoiceAgent(settings)
        await agent.initialize()
        
        try:
            tasks = [
                agent.process_text(f"Request {i}")
                for i in range(5)
            ]
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # All requests should complete successfully
            assert len(results) == 5
            assert all(isinstance(r, str) for r in results)
            
            # Should handle concurrency efficiently
            total_time = end_time - start_time
            assert total_time < 10.0
        finally:
            await agent.cleanup()
```

### Test Configuration

```python
# conftest.py
import pytest
import asyncio
from transcriber.config import Settings

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_settings():
    """Provide test configuration settings."""
    return Settings(
        agent=AgentSettings(
            model="llama3.2:1b",  # Faster model for testing
            temperature=0.1
        ),
        audio=AudioSettings(
            sample_rate=16000,
            chunk_duration=0.1
        ),
        performance=PerformanceSettings(
            monitoring_enabled=False  # Disable for tests
        )
    )

@pytest.fixture
async def mock_llm_service():
    """Provide mock LLM service for testing."""
    from unittest.mock import AsyncMock
    
    service = AsyncMock()
    service.generate_response.return_value = "Mock response"
    return service
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run with coverage
pytest --cov=transcriber --cov-report=html

# Run performance tests only
pytest -m performance

# Run tests in parallel
pytest -n auto

# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_config.py

# Run specific test function
pytest tests/unit/test_config.py::TestSettings::test_validation
```

## Pull Request Process

### PR Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Performance tests added/updated (if applicable)
- [ ] All tests pass locally

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Code is commented, particularly in hard-to-understand areas
- [ ] Documentation updated (if applicable)
- [ ] No new warnings introduced
- [ ] Related issues linked

## Related Issues
Fixes #(issue number)
Related to #(issue number)
```

### Review Process

1. **Automated Checks**
   - All tests must pass
   - Code coverage must not decrease
   - Linting checks must pass
   - Type checking must pass

2. **Manual Review**
   - Code quality and readability
   - Architecture and design decisions
   - Test coverage and quality
   - Documentation completeness

3. **Approval Requirements**
   - At least one maintainer approval
   - All conversations resolved
   - CI/CD pipeline passes

### Merge Requirements

- Branch is up to date with main
- All required checks pass
- No merge conflicts
- Squash and merge preferred for feature branches

## Issue Guidelines

### Bug Reports

```markdown
**Bug Description**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
A clear description of what you expected to happen.

**Environment**
- OS: [e.g. macOS 13.0]
- Python version: [e.g. 3.10.8]
- Project version: [e.g. 1.0.0]
- Audio device: [e.g. Built-in microphone]

**Additional Context**
Add any other context about the problem here.
```

### Feature Requests

```markdown
**Feature Description**
A clear description of what you want to happen.

**Use Case**
Describe the problem you're trying to solve.

**Proposed Solution**
Describe the solution you'd like.

**Alternatives Considered**
Describe any alternative solutions you've considered.

**Additional Context**
Add any other context or screenshots about the feature request here.
```

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `performance`: Performance-related issues
- `security`: Security-related issues
- `breaking change`: Changes that break backward compatibility

## Documentation Standards

### README Updates

When adding new features, update the main README.md:

- Add to feature list if it's user-facing
- Update installation instructions if needed
- Add to usage examples if appropriate

### API Documentation

For new APIs, add to [`docs/API.md`](API.md):

- Complete function/class documentation
- Usage examples
- Error handling information

### User Guide Updates

For user-facing features, update [`docs/USER_GUIDE.md`](USER_GUIDE.md):

- Step-by-step instructions
- Screenshots if helpful
- Common use cases

### Code Comments

```python
# Good comments explain WHY, not WHAT
def calculate_audio_features(audio_data: bytes) -> Dict[str, float]:
    """Calculate audio features for voice activity detection.
    
    We use spectral features because they're more robust to noise
    than simple amplitude-based detection.
    """
    # Convert to frequency domain for spectral analysis
    fft_data = np.fft.fft(audio_data)
    
    # Calculate spectral centroid (brightness indicator)
    spectral_centroid = np.sum(frequencies * magnitudes) / np.sum(magnitudes)
    
    return {"spectral_centroid": spectral_centroid}
```

## Release Process

### Version Numbering

We use Semantic Versioning (SemVer):

- `MAJOR.MINOR.PATCH` (e.g., 1.2.3)
- `MAJOR`: Breaking changes
- `MINOR`: New features (backward compatible)
- `PATCH`: Bug fixes (backward compatible)

### Release Checklist

1. **Pre-release**
   - [ ] All tests pass
   - [ ] Documentation updated
   - [ ] CHANGELOG.md updated
   - [ ] Version bumped in pyproject.toml

2. **Release**
   - [ ] Create release branch
   - [ ] Tag release
   - [ ] Build and test package
   - [ ] Publish to PyPI

3. **Post-release**
   - [ ] Update main branch
   - [ ] Create GitHub release
   - [ ] Announce release

### Changelog Format

```markdown
# Changelog

## [1.2.0] - 2024-01-15

### Added
- New voice activity detection algorithm
- Support for additional audio formats
- Performance monitoring dashboard

### Changed
- Improved response latency by 30%
- Updated default model to llama3.2:3b

### Fixed
- Fixed memory leak in audio processing
- Resolved session persistence issues

### Deprecated
- Old configuration format (will be removed in 2.0.0)

### Security
- Updated dependencies to address security vulnerabilities
```

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment:

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Respect different viewpoints and experiences

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussion
- **Pull Requests**: Code contributions and reviews

### Getting Help

1. **Check existing documentation**
   - README.md for basic usage
   - docs/ folder for detailed guides
   - API documentation for development

2. **Search existing issues**
   - Someone may have already reported the same issue
   - Look for similar feature requests

3. **Create a new issue**
   - Use appropriate templates
   - Provide detailed information
   - Be patient for responses

### Recognition

Contributors are recognized in:

- CONTRIBUTORS.md file
- Release notes for significant contributions
- GitHub contributor statistics

---

Thank you for contributing to the AI Voice Agent project! Your contributions help make this tool better for everyone.