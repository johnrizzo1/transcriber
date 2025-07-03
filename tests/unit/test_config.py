"""
Unit tests for configuration system.
"""

import os
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from transcriber.config import (
    AudioConfig,
    AgentConfig,
    VoiceConfig,
    WhisperConfig,
    Settings
)


@pytest.mark.unit
class TestAudioConfig:
    """Test AudioConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = AudioConfig()
        assert config.sample_rate == 16000
        assert config.chunk_duration == 0.1
        assert config.channels == 1
        assert config.input_device is None
        assert config.output_device is None
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = AudioConfig(
            sample_rate=22050,
            chunk_duration=0.2,
            channels=2,
            input_device=1,
            output_device=2
        )
        assert config.sample_rate == 22050
        assert config.chunk_duration == 0.2
        assert config.channels == 2
        assert config.input_device == 1
        assert config.output_device == 2
    
    def test_validation(self):
        """Test configuration validation."""
        # Valid values should work
        AudioConfig(sample_rate=16000, channels=1)
        
        # Invalid types should raise ValidationError
        with pytest.raises(ValidationError):
            AudioConfig(sample_rate="invalid")
        
        with pytest.raises(ValidationError):
            AudioConfig(channels="invalid")


@pytest.mark.unit
class TestAgentConfig:
    """Test AgentConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = AgentConfig()
        assert config.model == "llama3.2:3b"
        assert config.base_url == "http://localhost:11434"
        assert config.max_context_length == 4096
        assert config.temperature == 0.7
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = AgentConfig(
            model="llama3.2:1b",
            base_url="http://localhost:8080",
            max_context_length=2048,
            temperature=0.5
        )
        assert config.model == "llama3.2:1b"
        assert config.base_url == "http://localhost:8080"
        assert config.max_context_length == 2048
        assert config.temperature == 0.5
    
    def test_validation(self):
        """Test configuration validation."""
        # Valid values should work
        AgentConfig(temperature=0.0)
        AgentConfig(temperature=1.0)
        
        # Invalid types should raise ValidationError
        with pytest.raises(ValidationError):
            AgentConfig(temperature="invalid")
        
        with pytest.raises(ValidationError):
            AgentConfig(max_context_length="invalid")


@pytest.mark.unit
class TestVoiceConfig:
    """Test VoiceConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = VoiceConfig()
        assert config.vad_threshold == 0.5
        assert config.speech_timeout == 1.0
        assert config.max_speech_duration == 30.0
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = VoiceConfig(
            vad_threshold=0.7,
            speech_timeout=2.0,
            max_speech_duration=60.0
        )
        assert config.vad_threshold == 0.7
        assert config.speech_timeout == 2.0
        assert config.max_speech_duration == 60.0
    
    def test_validation(self):
        """Test configuration validation."""
        # Valid values should work
        VoiceConfig(vad_threshold=0.0)
        VoiceConfig(vad_threshold=1.0)
        
        # Invalid types should raise ValidationError
        with pytest.raises(ValidationError):
            VoiceConfig(vad_threshold="invalid")


@pytest.mark.unit
class TestWhisperConfig:
    """Test WhisperConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = WhisperConfig()
        assert config.model == "tiny"
        assert config.device == "cpu"
        assert config.compute_type == "int8"
        assert config.language is None
        assert config.beam_size == 5
        assert config.best_of == 5
        assert config.temperature == 0.0
        assert config.compression_ratio_threshold == 2.4
        assert config.log_prob_threshold == -1.0
        assert config.no_speech_threshold == 0.6
        assert config.condition_on_previous_text is True
        assert config.initial_prompt is None
        assert config.chunk_duration == 2.0
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = WhisperConfig(
            model="base",
            device="cuda",
            compute_type="float16",
            language="en",
            beam_size=10,
            temperature=0.1
        )
        assert config.model == "base"
        assert config.device == "cuda"
        assert config.compute_type == "float16"
        assert config.language == "en"
        assert config.beam_size == 10
        assert config.temperature == 0.1
    
    def test_validation(self):
        """Test configuration validation."""
        # Valid values should work
        WhisperConfig(model="tiny")
        WhisperConfig(model="base")
        
        # Invalid types should raise ValidationError
        with pytest.raises(ValidationError):
            WhisperConfig(beam_size="invalid")


@pytest.mark.unit
class TestSettings:
    """Test Settings class."""
    
    def test_default_initialization(self):
        """Test default settings initialization."""
        settings = Settings()
        
        # Check that nested configs are initialized
        assert isinstance(settings.audio, AudioConfig)
        assert isinstance(settings.agent, AgentConfig)
        assert isinstance(settings.voice, VoiceConfig)
        assert isinstance(settings.whisper, WhisperConfig)
        
        # Check default values
        assert settings.audio.sample_rate == 16000
        assert settings.agent.model == "llama3.2:3b"
        assert settings.voice.vad_threshold == 0.5
        assert settings.whisper.model == "tiny"
    
    def test_environment_variable_override(self):
        """Test environment variable configuration override."""
        # Set environment variables
        env_vars = {
            'TRANSCRIBER_AUDIO__SAMPLE_RATE': '22050',
            'TRANSCRIBER_AGENT__MODEL': 'llama3.2:1b',
            'TRANSCRIBER_VOICE__VAD_THRESHOLD': '0.7',
            'TRANSCRIBER_WHISPER__MODEL': 'base'
        }
        
        # Temporarily set environment variables
        original_env = {}
        for key, value in env_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            settings = Settings()
            
            # Check that environment variables override defaults
            assert settings.audio.sample_rate == 22050
            assert settings.agent.model == 'llama3.2:1b'
            assert settings.voice.vad_threshold == 0.7
            assert settings.whisper.model == 'base'
            
        finally:
            # Restore original environment
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
    
    def test_nested_config_access(self):
        """Test accessing nested configuration values."""
        settings = Settings()
        
        # Test direct access
        assert settings.audio.sample_rate == 16000
        assert settings.agent.base_url == "http://localhost:11434"
        
        # Test modification
        settings.audio.sample_rate = 48000
        assert settings.audio.sample_rate == 48000
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        settings = Settings()
        
        # Should be able to convert to dict
        config_dict = settings.model_dump()
        assert isinstance(config_dict, dict)
        assert 'audio' in config_dict
        assert 'agent' in config_dict
        assert 'voice' in config_dict
        assert 'whisper' in config_dict
        
        # Check nested structure
        assert config_dict['audio']['sample_rate'] == 16000
        assert config_dict['agent']['model'] == "llama3.2:3b"
    
    def test_config_validation_errors(self):
        """Test configuration validation with invalid values."""
        settings = Settings()
        
        # Invalid audio sample rate
        with pytest.raises(ValidationError):
            settings.audio.sample_rate = "invalid"
        
        # Invalid agent temperature
        with pytest.raises(ValidationError):
            settings.agent.temperature = "invalid"
    
    def test_data_dir_configuration(self):
        """Test data directory configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings()
            settings.data_dir = temp_dir
            
            assert settings.data_dir == temp_dir
            assert Path(settings.data_dir).exists()
    
    def test_session_configuration(self):
        """Test session-related configuration."""
        settings = Settings()
        
        # Check if session config exists (if implemented)
        if hasattr(settings, 'session'):
            assert hasattr(settings.session, 'enabled')
    
    def test_performance_configuration(self):
        """Test performance-related configuration."""
        settings = Settings()
        
        # Check if performance config exists (if implemented)
        if hasattr(settings, 'performance'):
            assert hasattr(settings.performance, 'enabled')
    
    def test_tools_configuration(self):
        """Test tools-related configuration."""
        settings = Settings()
        
        # Check if tools config exists (if implemented)
        if hasattr(settings, 'tools'):
            assert hasattr(settings.tools, 'enabled')


@pytest.mark.unit
class TestConfigurationIntegration:
    """Test configuration system integration."""
    
    def test_complete_configuration_flow(self):
        """Test complete configuration initialization and usage."""
        # Create settings with custom values
        settings = Settings()
        settings.audio.sample_rate = 22050
        settings.agent.model = "llama3.2:1b"
        settings.voice.vad_threshold = 0.8
        settings.whisper.model = "base"
        
        # Verify all values are set correctly
        assert settings.audio.sample_rate == 22050
        assert settings.agent.model == "llama3.2:1b"
        assert settings.voice.vad_threshold == 0.8
        assert settings.whisper.model == "base"
        
        # Test serialization and deserialization
        config_dict = settings.model_dump()
        new_settings = Settings(**config_dict)
        
        assert new_settings.audio.sample_rate == 22050
        assert new_settings.agent.model == "llama3.2:1b"
        assert new_settings.voice.vad_threshold == 0.8
        assert new_settings.whisper.model == "base"
    
    def test_configuration_with_missing_optional_fields(self):
        """Test configuration with missing optional fields."""
        # Create minimal configuration
        minimal_config = {
            'audio': {'sample_rate': 16000},
            'agent': {'model': 'llama3.2:3b'},
            'voice': {'vad_threshold': 0.5},
            'whisper': {'model': 'tiny'}
        }
        
        settings = Settings(**minimal_config)
        
        # Should use defaults for missing fields
        assert settings.audio.channels == 1  # default
        assert settings.agent.temperature == 0.7  # default
        assert settings.voice.speech_timeout == 1.0  # default
        assert settings.whisper.beam_size == 5  # default