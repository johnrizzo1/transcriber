"""
Pytest configuration and shared fixtures for the AI Voice Agent test suite.
"""

import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    import numpy as np
except ImportError:
    np = None

from transcriber.config import Settings
from transcriber.session.manager import SessionManager
from transcriber.session.storage import SessionStorage
from transcriber.tools.registry import ToolRegistry
from transcriber.performance.integration import PerformanceIntegration


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test data."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def test_settings(temp_dir: Path) -> Settings:
    """Create test settings with temporary data directory."""
    settings = Settings()
    settings.data_dir = str(temp_dir)
    settings.session.enabled = True
    settings.session.auto_start_session = True
    
    # Use smaller models for testing
    settings.agent.model = "llama3.2:1b"
    settings.whisper.model = "tiny"
    
    # Disable audio devices for testing
    settings.audio.input_device = None
    settings.audio.output_device = None
    
    return settings


@pytest.fixture
async def session_manager(
    test_settings: Settings
) -> AsyncGenerator[SessionManager, None]:
    """Create a session manager for testing."""
    manager = SessionManager(data_dir=test_settings.data_dir)
    await manager.initialize()
    yield manager
    await manager.close()


@pytest.fixture
async def session_storage(
    temp_dir: Path
) -> AsyncGenerator[SessionStorage, None]:
    """Create session storage for testing."""
    db_path = temp_dir / "test_sessions.db"
    storage = SessionStorage(str(db_path))
    await storage.initialize()
    yield storage
    await storage.close()


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Create a clean tool registry for testing."""
    registry = ToolRegistry()
    return registry


@pytest.fixture
async def performance_integration(
    temp_dir: Path
) -> AsyncGenerator[PerformanceIntegration, None]:
    """Create performance integration for testing."""
    perf = PerformanceIntegration(data_dir=str(temp_dir), enable_all=True)
    await perf.initialize()
    yield perf
    await perf.cleanup()


@pytest.fixture
def mock_audio_data():
    """Generate mock audio data for testing."""
    if np is None:
        # Fallback to simple list for testing without numpy
        sample_rate = 16000
        duration = 1.0
        return [0.0] * int(sample_rate * duration)
    
    # Generate 1 second of sine wave at 440Hz
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio


@pytest.fixture
def mock_audio_chunk():
    """Generate a small mock audio chunk for testing."""
    if np is None:
        # Fallback to simple list for testing without numpy
        sample_rate = 16000
        duration = 0.1
        return [0.0] * int(sample_rate * duration)
    
    # Generate 100ms of audio
    sample_rate = 16000
    duration = 0.1
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio


@pytest.fixture
def mock_silence_data():
    """Generate mock silence data for testing."""
    if np is None:
        # Fallback to simple list for testing without numpy
        sample_rate = 16000
        duration = 1.0
        return [0.0] * int(sample_rate * duration)
    
    sample_rate = 16000
    duration = 1.0
    return np.zeros(int(sample_rate * duration), dtype=np.float32)


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client for testing."""
    with patch('transcriber.agent.llm.ollama.AsyncClient') as mock_client:
        mock_instance = AsyncMock()
        mock_client.return_value = mock_instance
        
        # Mock chat response
        mock_instance.chat.return_value = {
            'message': {
                'content': 'Test response from LLM',
                'role': 'assistant'
            }
        }
        
        # Mock streaming chat response
        async def mock_chat_stream(*args, **kwargs):
            yield {'message': {'content': 'Test ', 'role': 'assistant'}}
            yield {'message': {'content': 'streaming ', 'role': 'assistant'}}
            yield {'message': {'content': 'response', 'role': 'assistant'}}
        
        mock_instance.chat.return_value = mock_chat_stream()
        
        yield mock_instance


@pytest.fixture
def mock_whisper_model():
    """Mock Whisper model for testing."""
    with patch('transcriber.audio.whisper_stt.WhisperModel') as mock_model:
        mock_instance = MagicMock()
        mock_model.return_value = mock_instance
        
        # Mock transcribe method
        mock_instance.transcribe.return_value = (
            [{'text': 'This is a test transcription'}],
            {'language': 'en', 'language_probability': 0.99}
        )
        
        yield mock_instance


@pytest.fixture
def mock_vad():
    """Mock WebRTC VAD for testing."""
    with patch('transcriber.audio.vad.webrtcvad.Vad') as mock_vad:
        mock_instance = MagicMock()
        mock_vad.return_value = mock_instance
        
        # Mock is_speech method - alternate between speech and silence
        mock_instance.is_speech.side_effect = lambda *args: True
        
        yield mock_instance


@pytest.fixture
def mock_sounddevice():
    """Mock sounddevice for testing."""
    with patch('transcriber.audio.capture.sd') as mock_sd:
        mock_sd.query_devices.return_value = [
            {
                'name': 'Test Input Device',
                'max_input_channels': 1,
                'default_samplerate': 16000
            },
            {
                'name': 'Test Output Device',
                'max_output_channels': 2,
                'default_samplerate': 16000
            }
        ]
        
        mock_sd.default.device = [0, 1]
        mock_sd.default.samplerate = 16000
        mock_sd.default.channels = 1
        
        yield mock_sd


@pytest.fixture
def mock_edge_tts():
    """Mock Edge TTS for testing."""
    with patch('transcriber.audio.edge_tts.edge_tts') as mock_tts:
        mock_communicate = AsyncMock()
        mock_tts.Communicate.return_value = mock_communicate
        
        # Mock TTS output
        async def mock_stream():
            yield {'type': 'audio', 'data': b'fake_audio_data'}
        
        mock_communicate.stream.return_value = mock_stream()
        
        yield mock_tts


# Test data fixtures
@pytest.fixture
def sample_session_data():
    """Sample session data for testing."""
    return {
        'title': 'Test Session',
        'messages': [
            {
                'content': 'Hello',
                'type': 'user',
                'timestamp': '2024-01-01T12:00:00'
            },
            {
                'content': 'Hi there!',
                'type': 'assistant',
                'timestamp': '2024-01-01T12:00:01'
            },
        ]
    }


@pytest.fixture
def sample_tool_params():
    """Sample tool parameters for testing."""
    return {
        'calculator': {'expression': '2 + 2'},
        'file_ops': {'operation': 'list', 'path': '/tmp'},
        'system_info': {'info_type': 'cpu'},
        'text_processing': {'operation': 'uppercase', 'text': 'hello world'}
    }


# Performance test fixtures
@pytest.fixture
def performance_thresholds():
    """Performance thresholds for testing."""
    return {
        'vad_processing': 50,  # ms
        'stt_processing': 200,  # ms
        'llm_processing': 500,  # ms
        'tts_processing': 150,  # ms
        'end_to_end': 1000,  # ms
    }


# Async test helpers
@pytest.fixture
def async_timeout():
    """Default timeout for async tests."""
    return 10.0  # seconds


# Mock external services
@pytest.fixture
def mock_external_services(
    mock_ollama_client,
    mock_whisper_model,
    mock_vad,
    mock_sounddevice,
    mock_edge_tts
):
    """Combine all external service mocks."""
    return {
        'ollama': mock_ollama_client,
        'whisper': mock_whisper_model,
        'vad': mock_vad,
        'sounddevice': mock_sounddevice,
        'edge_tts': mock_edge_tts
    }


# Test markers are defined in pytest.ini