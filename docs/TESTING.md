# Testing Guide

Comprehensive guide to testing the AI Voice Agent project, including unit tests, integration tests, and performance testing.

## Table of Contents

1. [Overview](#overview)
2. [Test Structure](#test-structure)
3. [Running Tests](#running-tests)
4. [Unit Testing](#unit-testing)
5. [Integration Testing](#integration-testing)
6. [Performance Testing](#performance-testing)
7. [Test Coverage](#test-coverage)
8. [Mocking and Fixtures](#mocking-and-fixtures)
9. [Continuous Integration](#continuous-integration)
10. [Writing New Tests](#writing-new-tests)
11. [Debugging Tests](#debugging-tests)
12. [Best Practices](#best-practices)

## Overview

The AI Voice Agent uses a comprehensive testing strategy to ensure reliability, performance, and maintainability. Our testing approach includes:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions and workflows
- **Performance Tests**: Validate performance requirements and benchmarks
- **End-to-End Tests**: Test complete user workflows

### Testing Framework

We use **pytest** as our primary testing framework with several plugins:

```bash
pytest>=7.0.0              # Main testing framework
pytest-asyncio>=0.21.0     # Async test support
pytest-cov>=4.0.0          # Coverage reporting
pytest-mock>=3.10.0        # Enhanced mocking
pytest-xdist>=3.0.0        # Parallel test execution
pytest-benchmark>=4.0.0    # Performance benchmarking
```

## Test Structure

```
tests/
├── conftest.py                 # Shared test configuration and fixtures
├── unit/                       # Unit tests for individual components
│   ├── __init__.py
│   ├── test_audio_vad.py      # Voice activity detection tests
│   ├── test_config.py         # Configuration tests
│   ├── test_session_models.py # Session model tests
│   ├── test_tools_base.py     # Tool system base tests
│   └── test_tools_registry.py # Tool registry tests
├── integration/                # Integration tests
│   ├── __init__.py
│   ├── test_voice_pipeline.py # Complete voice pipeline tests
│   └── test_session_management.py # Session management tests
├── performance/                # Performance and benchmark tests
│   ├── __init__.py
│   ├── README.md              # Performance testing documentation
│   ├── test_benchmarks.py     # System benchmarks
│   ├── test_config.py         # Performance configuration tests
│   ├── test_simple.py         # Simple performance tests
│   └── test_stress.py         # Stress testing
└── run_performance_tests.py    # Performance test runner
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_config.py

# Run specific test function
pytest tests/unit/test_config.py::TestSettings::test_validation

# Run tests matching pattern
pytest -k "test_audio"

# Run tests by marker
pytest -m "unit"
pytest -m "integration"
pytest -m "performance"
```

### Parallel Execution

```bash
# Run tests in parallel (auto-detect CPU cores)
pytest -n auto

# Run tests with specific number of workers
pytest -n 4

# Run tests in parallel with coverage
pytest -n auto --cov=transcriber
```

### Coverage Reporting

```bash
# Run tests with coverage
pytest --cov=transcriber

# Generate HTML coverage report
pytest --cov=transcriber --cov-report=html

# Generate XML coverage report (for CI)
pytest --cov=transcriber --cov-report=xml

# Show missing lines
pytest --cov=transcriber --cov-report=term-missing

# Set minimum coverage threshold
pytest --cov=transcriber --cov-fail-under=80
```

### Test Categories

```bash
# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run only performance tests
pytest tests/performance/

# Run tests by custom markers
pytest -m "slow"      # Slow tests
pytest -m "fast"      # Fast tests
pytest -m "network"   # Tests requiring network
```

## Unit Testing

Unit tests focus on testing individual components in isolation.

### Example Unit Test

```python
import pytest
from unittest.mock import Mock, patch, AsyncMock
from transcriber.audio.vad import VADProcessor
from transcriber.config import AudioSettings

class TestVADProcessor:
    """Test suite for Voice Activity Detection processor."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.settings = AudioSettings(
            sample_rate=16000,
            chunk_duration=0.02,
            vad_aggressiveness=1
        )
        self.vad = VADProcessor(self.settings)
    
    def teardown_method(self):
        """Clean up after each test method."""
        # Clean up resources if needed
        pass
    
    def test_initialization(self):
        """Test VAD processor initialization."""
        assert self.vad.sample_rate == 16000
        assert self.vad.frame_duration == 0.02
        assert self.vad.aggressiveness == 1
    
    def test_initialization_with_invalid_settings(self):
        """Test initialization with invalid settings."""
        invalid_settings = AudioSettings(sample_rate=8000)  # Unsupported rate
        
        with pytest.raises(ValueError, match="Unsupported sample rate"):
            VADProcessor(invalid_settings)
    
    @patch('transcriber.audio.vad.webrtcvad.Vad')
    def test_process_chunk_with_speech(self, mock_vad_class):
        """Test processing audio chunk containing speech."""
        # Setup mock
        mock_vad_instance = Mock()
        mock_vad_instance.is_speech.return_value = True
        mock_vad_class.return_value = mock_vad_instance
        
        # Create test data
        audio_chunk = b'\x00\x01' * 160  # 320 bytes for 20ms at 16kHz
        
        # Test
        result = self.vad.process_chunk(audio_chunk)
        
        # Assertions
        assert result is True
        mock_vad_instance.is_speech.assert_called_once_with(16000, audio_chunk)
    
    @patch('transcriber.audio.vad.webrtcvad.Vad')
    def test_process_chunk_without_speech(self, mock_vad_class):
        """Test processing audio chunk without speech."""
        mock_vad_instance = Mock()
        mock_vad_instance.is_speech.return_value = False
        mock_vad_class.return_value = mock_vad_instance
        
        audio_chunk = b'\x00' * 320
        result = self.vad.process_chunk(audio_chunk)
        
        assert result is False
    
    def test_process_chunk_invalid_size(self):
        """Test processing chunk with invalid size."""
        invalid_chunk = b'\x00' * 100  # Too small
        
        with pytest.raises(ValueError, match="Invalid chunk size"):
            self.vad.process_chunk(invalid_chunk)
    
    @pytest.mark.parametrize("aggressiveness,expected", [
        (0, False),  # Least aggressive
        (1, True),   # Default
        (2, True),   # More aggressive
        (3, True),   # Most aggressive
    ])
    def test_different_aggressiveness_levels(self, aggressiveness, expected):
        """Test VAD with different aggressiveness levels."""
        settings = AudioSettings(vad_aggressiveness=aggressiveness)
        vad = VADProcessor(settings)
        
        # This would need actual audio data or more sophisticated mocking
        assert vad.aggressiveness == aggressiveness
```

### Async Unit Tests

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from transcriber.agent.llm import LLMService
from transcriber.config import AgentSettings

class TestLLMService:
    """Test suite for LLM service."""
    
    @pytest.fixture
    async def llm_service(self):
        """Create LLM service for testing."""
        settings = AgentSettings(
            model="llama3.2:1b",
            temperature=0.1,
            max_tokens=100
        )
        service = LLMService(settings)
        await service.initialize()
        yield service
        await service.cleanup()
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test LLM service initialization."""
        settings = AgentSettings(model="llama3.2:1b")
        service = LLMService(settings)
        
        await service.initialize()
        
        assert service.model == "llama3.2:1b"
        assert service.client is not None
        
        await service.cleanup()
    
    @pytest.mark.asyncio
    async def test_generate_response(self, llm_service):
        """Test response generation."""
        messages = [
            {"role": "user", "content": "What is 2 + 2?"}
        ]
        
        response = await llm_service.generate_response(messages)
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "4" in response
    
    @pytest.mark.asyncio
    async def test_generate_response_with_timeout(self, llm_service):
        """Test response generation with timeout."""
        messages = [
            {"role": "user", "content": "Very complex question..."}
        ]
        
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                llm_service.generate_response(messages),
                timeout=0.001  # Very short timeout
            )
    
    @pytest.mark.asyncio
    @patch('transcriber.agent.llm.ollama.AsyncClient')
    async def test_generate_response_with_mock(self, mock_client_class):
        """Test response generation with mocked client."""
        # Setup mock
        mock_client = AsyncMock()
        mock_client.chat.return_value = {
            'message': {'content': 'Mocked response'}
        }
        mock_client_class.return_value = mock_client
        
        # Test
        settings = AgentSettings(model="test-model")
        service = LLMService(settings)
        await service.initialize()
        
        messages = [{"role": "user", "content": "Test"}]
        response = await service.generate_response(messages)
        
        assert response == "Mocked response"
        mock_client.chat.assert_called_once()
        
        await service.cleanup()
```

### Testing Tools

```python
import pytest
from unittest.mock import Mock, patch
from transcriber.tools.builtin.calculator import CalculatorTool
from transcriber.tools.base import ToolExecutionError, ToolValidationError

class TestCalculatorTool:
    """Test suite for calculator tool."""
    
    def setup_method(self):
        """Set up calculator tool for testing."""
        self.tool = CalculatorTool()
    
    def test_metadata(self):
        """Test tool metadata."""
        metadata = self.tool.metadata
        
        assert metadata.name == "calculator"
        assert "calculate" in metadata.description.lower()
        assert len(metadata.examples) > 0
    
    def test_parameters(self):
        """Test tool parameters."""
        params = self.tool.parameters
        
        assert len(params) == 1
        expression_param = params[0]
        assert expression_param.name == "expression"
        assert expression_param.required is True
    
    @pytest.mark.asyncio
    async def test_simple_calculation(self):
        """Test simple mathematical calculation."""
        result = await self.tool.execute(expression="2 + 2")
        
        assert "4" in str(result)
    
    @pytest.mark.asyncio
    async def test_complex_calculation(self):
        """Test complex mathematical calculation."""
        result = await self.tool.execute(expression="(10 + 5) * 2 / 3")
        
        assert "10" in str(result)  # (15 * 2) / 3 = 10
    
    @pytest.mark.asyncio
    async def test_invalid_expression(self):
        """Test handling of invalid mathematical expression."""
        with pytest.raises(ToolExecutionError):
            await self.tool.execute(expression="2 + + 2")
    
    @pytest.mark.asyncio
    async def test_dangerous_expression(self):
        """Test handling of potentially dangerous expressions."""
        with pytest.raises(ToolExecutionError):
            await self.tool.execute(expression="__import__('os').system('ls')")
    
    @pytest.mark.asyncio
    async def test_missing_parameter(self):
        """Test handling of missing required parameter."""
        with pytest.raises(ToolValidationError):
            await self.tool.execute()  # No expression provided
    
    @pytest.mark.parametrize("expression,expected", [
        ("1 + 1", "2"),
        ("10 - 3", "7"),
        ("4 * 5", "20"),
        ("15 / 3", "5"),
        ("2 ** 3", "8"),
        ("sqrt(16)", "4"),
        ("sin(0)", "0"),
    ])
    @pytest.mark.asyncio
    async def test_various_expressions(self, expression, expected):
        """Test various mathematical expressions."""
        result = await self.tool.execute(expression=expression)
        assert expected in str(result)
```

## Integration Testing

Integration tests verify that components work together correctly.

### Voice Pipeline Integration

```python
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from transcriber.agent.core import VoiceAgent
from transcriber.config import Settings

class TestVoicePipelineIntegration:
    """Integration tests for the complete voice pipeline."""
    
    @pytest.fixture
    async def voice_agent(self):
        """Create and initialize voice agent for testing."""
        settings = Settings()
        # Use test-specific settings
        settings.agent.model = "llama3.2:1b"  # Faster model for testing
        settings.audio.sample_rate = 16000
        
        agent = VoiceAgent(settings)
        await agent.initialize()
        yield agent
        await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_text_processing_flow(self, voice_agent):
        """Test complete text processing flow."""
        user_input = "What is 2 + 2?"
        
        response = await voice_agent.process_text(user_input)
        
        assert isinstance(response, str)
        assert len(response) > 0
        # Should contain the answer
        assert "4" in response
    
    @pytest.mark.asyncio
    async def test_tool_execution_flow(self, voice_agent):
        """Test tool execution through the agent."""
        user_input = "Calculate 10 + 15"
        
        response = await voice_agent.process_text(user_input)
        
        assert isinstance(response, str)
        assert "25" in response
    
    @pytest.mark.asyncio
    @patch('transcriber.audio.stt.WhisperSTT.transcribe')
    async def test_audio_processing_flow(self, mock_transcribe, voice_agent):
        """Test audio processing flow with mocked STT."""
        # Mock STT to return known text
        mock_transcribe.return_value = "What is the weather like?"
        
        # Create mock audio data
        audio_data = b'\x00\x01' * 8000  # 1 second of mock audio
        
        response = await voice_agent.process_audio(audio_data)
        
        assert isinstance(response, str)
        assert len(response) > 0
        mock_transcribe.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_session_integration(self, voice_agent):
        """Test session management integration."""
        # Start a session
        session = await voice_agent.session_manager.create_session("Test Session")
        
        # Process some messages
        response1 = await voice_agent.process_text("Hello")
        response2 = await voice_agent.process_text("What's 5 + 5?")
        
        # Verify session contains messages
        messages = await voice_agent.session_manager.get_session_messages(session.id)
        
        assert len(messages) >= 4  # 2 user messages + 2 assistant responses
        assert any("Hello" in msg.content for msg in messages)
        assert any("10" in msg.content for msg in messages)
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, voice_agent):
        """Test error handling across the pipeline."""
        # Test with input that might cause tool errors
        user_input = "Calculate invalid_expression"
        
        response = await voice_agent.process_text(user_input)
        
        # Should handle error gracefully
        assert isinstance(response, str)
        assert len(response) > 0
        # Should indicate error without crashing
        assert any(word in response.lower() for word in ["error", "sorry", "unable"])
```

### Session Management Integration

```python
import pytest
from uuid import uuid4
from transcriber.session.manager import SessionManager
from transcriber.session.models import Message, MessageType, SessionStatus

class TestSessionManagementIntegration:
    """Integration tests for session management."""
    
    @pytest.fixture
    async def session_manager(self):
        """Create session manager for testing."""
        manager = SessionManager()
        await manager.initialize()
        yield manager
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_complete_session_workflow(self, session_manager):
        """Test complete session workflow."""
        # Create session
        session = await session_manager.create_session("Test Workflow")
        assert session.title == "Test Workflow"
        assert session.status == SessionStatus.ACTIVE
        
        # Add messages
        user_msg = Message(
            session_id=session.id,
            content="Hello, AI!",
            message_type=MessageType.USER
        )
        await session_manager.save_message(session.id, user_msg)
        
        assistant_msg = Message(
            session_id=session.id,
            content="Hello! How can I help you?",
            message_type=MessageType.ASSISTANT
        )
        await session_manager.save_message(session.id, assistant_msg)
        
        # Retrieve messages
        messages = await session_manager.get_session_messages(session.id)
        assert len(messages) == 2
        assert messages[0].content == "Hello, AI!"
        assert messages[1].content == "Hello! How can I help you?"
        
        # Export session
        export_data = await session_manager.export_session(session.id)
        assert "Hello, AI!" in export_data
        assert "Hello! How can I help you?" in export_data
        
        # Complete session
        await session_manager.complete_session(session.id)
        updated_session = await session_manager.get_session(session.id)
        assert updated_session.status == SessionStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_concurrent_sessions(self, session_manager):
        """Test handling multiple concurrent sessions."""
        # Create multiple sessions
        sessions = []
        for i in range(5):
            session = await session_manager.create_session(f"Session {i}")
            sessions.append(session)
        
        # Add messages to each session concurrently
        import asyncio
        
        async def add_messages_to_session(session):
            for j in range(3):
                msg = Message(
                    session_id=session.id,
                    content=f"Message {j} in {session.title}",
                    message_type=MessageType.USER
                )
                await session_manager.save_message(session.id, msg)
        
        # Run concurrently
        await asyncio.gather(*[
            add_messages_to_session(session) for session in sessions
        ])
        
        # Verify all sessions have correct messages
        for session in sessions:
            messages = await session_manager.get_session_messages(session.id)
            assert len(messages) == 3
            assert all(session.title.split()[-1] in msg.content for msg in messages)
```

## Performance Testing

Performance tests validate system performance and identify bottlenecks.

### Basic Performance Tests

```python
import pytest
import time
import asyncio
import psutil
from transcriber.agent.core import VoiceAgent
from transcriber.performance.benchmarks import BenchmarkRunner

class TestPerformance:
    """Performance tests for the voice agent system."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_response_latency(self):
        """Test response latency is within acceptable limits."""
        settings = Settings()
        settings.agent.model = "llama3.2:1b"  # Fast model for testing
        
        agent = VoiceAgent(settings)
        await agent.initialize()
        
        try:
            # Measure response time
            start_time = time.time()
            response = await agent.process_text("Hello")
            end_time = time.time()
            
            latency = end_time - start_time
            
            # Assert reasonable response time
            assert latency < 3.0, f"Response took {latency:.2f}s, expected < 3.0s"
            assert len(response) > 0
            
        finally:
            await agent.cleanup()
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        settings = Settings()
        agent = VoiceAgent(settings)
        await agent.initialize()
        
        try:
            # Create multiple concurrent requests
            requests = [
                agent.process_text(f"Calculate {i} + {i}")
                for i in range(1, 6)
            ]
            
            start_time = time.time()
            responses = await asyncio.gather(*requests)
            end_time = time.time()
            
            total_time = end_time - start_time
            
            # All requests should complete successfully
            assert len(responses) == 5
            assert all(isinstance(r, str) and len(r) > 0 for r in responses)
            
            # Should handle concurrency efficiently
            assert total_time < 15.0, f"Concurrent requests took {total_time:.2f}s"
            
        finally:
            await agent.cleanup()
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory usage during operation."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        settings = Settings()
        agent = VoiceAgent(settings)
        await agent.initialize()
        
        try:
            # Process multiple requests
            for i in range(10):
                await agent.process_text(f"Test message {i}")
            
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            # Memory usage should be reasonable
            assert memory_increase < 500, f"Memory increased by {memory_increase:.1f}MB"
            
        finally:
            await agent.cleanup()
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_benchmark_suite(self):
        """Run the complete benchmark suite."""
        settings = Settings()
        runner = BenchmarkRunner(settings)
        
        results = await runner.run_all_benchmarks()
        
        # Verify benchmark results
        assert results.latency_benchmark.avg_latency < 2.0
        assert results.throughput_benchmark.requests_per_second > 1.0
        assert results.memory_benchmark.peak_memory_mb < 1000
        
        # Print results for manual inspection
        print(f"Average latency: {results.latency_benchmark.avg_latency:.2f}s")
        print(f"Throughput: {results.throughput_benchmark.requests_per_second:.1f} req/s")
        print(f"Peak memory: {results.memory_benchmark.peak_memory_mb:.1f}MB")
```

### Stress Testing

```python
import pytest
import asyncio
import time
from transcriber.agent.core import VoiceAgent

class TestStress:
    """Stress tests for system reliability."""
    
    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_sustained_load(self):
        """Test system under sustained load."""
        settings = Settings()
        agent = VoiceAgent(settings)
        await agent.initialize()
        
        try:
            # Run sustained load for 60 seconds
            end_time = time.time() + 60
            request_count = 0
            errors = 0
            
            while time.time() < end_time:
                try:
                    await agent.process_text(f"Request {request_count}")
                    request_count += 1
                except Exception as e:
                    errors += 1
                    print(f"Error in request {request_count}: {e}")
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)
            
            # Should handle most requests successfully
            error_rate = errors / request_count if request_count > 0 else 1
            assert error_rate < 0.05, f"Error rate {error_rate:.2%} too high"
            assert request_count > 100, f"Only processed {request_count} requests"
            
        finally:
            await agent.cleanup()
    
    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self):
        """Test for memory leaks during extended operation."""
        import gc
        import psutil
        
        process = psutil.Process()
        
        settings = Settings()
        agent = VoiceAgent(settings)
        await agent.initialize()
        
        try:
            memory_samples = []
            
            for i in range(100):
                await agent.process_text(f"Memory test {i}")
                
                if i % 10 == 0:  # Sample every 10 requests
                    gc.collect()  # Force garbage collection
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    memory_samples.append(memory_mb)
            
            # Check for memory growth trend
            if len(memory_samples) > 2:
                initial_memory = memory_samples[0]
                final_memory = memory_samples[-1]
                memory_growth = final_memory - initial_memory
                
                # Allow some growth but not excessive
                assert memory_growth < 100, f"Memory grew by {memory_growth:.1f}MB"
            
        finally:
            await agent.cleanup()
```

## Test Coverage

### Coverage Configuration

```ini
# setup.cfg
[coverage:run]
source = transcriber
omit = 
    */tests/*
    */venv/*
    */__pycache__/*
    */migrations/*
    */settings/*

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    class .*\(Protocol\):
    @(abc\.)?abstractmethod

[coverage:html]
directory = htmlcov
```

### Coverage Targets

- **Overall Coverage**: Minimum 80%
- **Core Components**: Minimum 90%
  - `transcriber/agent/`
  - `transcriber/audio/`
  - `transcriber/tools/`
- **Configuration**: Minimum 85%
- **Utilities**: Minimum 75%

### Measuring Coverage

```bash
# Generate coverage report
pytest --cov=transcriber --cov-report=html --cov-report=term

# Check coverage for specific modules
pytest --cov=transcriber.agent --cov-report=term-missing

# Fail if coverage below threshold
pytest --cov=transcriber --cov-fail-under=80

# Generate coverage badge
coverage-badge -o coverage.svg
```

## Mocking and Fixtures

### Common Fixtures

```python
# conftest.py
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from transcriber.config import Settings, AgentSettings, AudioSettings

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_settings():
    """Provide test configuration settings."""
    return Settings(
        agent=AgentSettings(
            model="llama3.2:1b",
            temperature=0.1,
            max_tokens=100
        ),
        audio=AudioSettings(
            sample_rate=16000,
            chunk_duration=0.02
        )
    )

@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing."""
    service = AsyncMock()
    service.generate_response.return_value = "Mock response"
    service.generate_response_stream.return_value = async_generator(["Mock", " response"])
    return service

@pytest.fixture
def mock_audio_data():
    """Generate mock audio data for testing."""
    # 1 second of 16kHz mono audio
    return b'\x00\x01' * 8000

@pytest.fixture
async def temp_database():
    """Create temporary database for testing."""
    import tempfile
    import os
    
    # Create temporary database file
    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(db_fd)
    
    yield db_path
    
    # Clean up
    if os.path.exists(db_path):
        os.unlink(db_path)

async def async_generator(items):
    """Helper for creating async generators in tests."""
    for item in items:
        yield item
```

### Mocking External Services

```python
import pytest
from unittest.mock import patch, Mock, AsyncMock

class TestWithMocks:
    """Example of testing with various mocking strategies."""
    
    @patch('transcriber.audio.stt.whisper.load_model')
    def test_whisper_model_loading(self, mock_load_model):
        """Test Whisper model loading with mock."""
        mock_model = Mock()
        mock_model.transcribe.return_value = {"text": "Hello world"}
        mock_load_model.return_value = mock_model
        
        from transcriber.audio.whisper_stt import WhisperSTT
        stt = WhisperSTT()
        
        result = stt.transcribe(b"mock_audio_data")
        assert result == "Hello world"
        mock_load_model.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_http_request_mock(self, mock_get):
        """Test HTTP requests with aiohttp mock."""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = "Mock response"
        mock_get.return_value.__aenter__.return_value = mock_response
        
        # Test code that makes HTTP request
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get("http://example.com") as response:
                text = await response.text()
                assert text == "Mock response"
                assert response.status == 200
    
    @patch.dict('os.environ', {'TRANSCRIBER_MODEL': 'test-model'})
    def test_environment_variable_mock(self):
        