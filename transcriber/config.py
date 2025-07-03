"""
Configuration and settings for the Transcriber AI Voice Agent.
"""

from typing import Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class AudioConfig(BaseModel):
    """Audio system configuration."""
    sample_rate: int = 16000
    chunk_duration: float = 0.1  # seconds
    channels: int = 1
    input_device: Optional[int] = None
    output_device: Optional[int] = None


class AgentConfig(BaseModel):
    """AI agent configuration."""
    model: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434"
    max_context_length: int = 4096
    temperature: float = 0.7


class VoiceConfig(BaseModel):
    """Voice processing configuration."""
    vad_threshold: float = 0.5
    speech_timeout: float = 1.0  # seconds of silence before processing
    max_speech_duration: float = 30.0  # seconds


class WhisperConfig(BaseModel):
    """Whisper STT configuration."""
    model: str = "tiny"  # tiny, base, small, medium, large
    device: str = "cpu"  # cpu, cuda, auto
    compute_type: str = "int8"  # int8, int16, float16, float32
    language: Optional[str] = None  # auto-detect if None
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = True
    initial_prompt: Optional[str] = None
    chunk_duration: float = 2.0  # seconds for streaming chunks


class PiperConfig(BaseModel):
    """Piper TTS configuration."""
    model: str = "en_US-lessac-medium"  # Voice model to use
    models_dir: str = "./models/piper"  # Directory to store voice models
    sample_rate: int = 22050  # TTS output sample rate
    speed: float = 1.0  # Speech speed multiplier
    volume: float = 1.0  # Volume multiplier
    sentence_silence: float = 0.2  # Silence between sentences (seconds)


class SessionConfig(BaseModel):
    """Session management configuration."""
    enabled: bool = True
    db_path: Optional[str] = None  # Auto-generated if None
    auto_start_session: bool = True
    session_timeout_minutes: int = 60  # Auto-complete after inactivity
    max_messages_per_session: int = 1000
    cleanup_days: int = 30  # Clean up sessions older than this
    export_formats: list[str] = ["json", "txt"]
    backup_enabled: bool = True
    backup_interval_hours: int = 24


class PerformanceConfig(BaseModel):
    """Performance monitoring and optimization configuration."""
    monitoring_enabled: bool = True
    monitoring_interval: float = 1.0  # seconds
    profiling_enabled: bool = False
    benchmarking_enabled: bool = False
    auto_optimization: bool = False
    memory_limit_mb: Optional[int] = None  # Memory usage limit
    cleanup_interval_minutes: int = 60  # Automatic cleanup interval
    metrics_retention_days: int = 7  # How long to keep performance metrics
    export_metrics: bool = True
    alert_thresholds: dict = {
        "memory_percent": 85.0,  # Alert when memory usage exceeds this
        "cpu_percent": 90.0,     # Alert when CPU usage exceeds this
        "latency_ms": 1000.0     # Alert when latency exceeds this
    }


class MemoryConfig(BaseModel):
    """Vector memory configuration for persistent context."""
    enabled: bool = True
    chromadb_path: str = "./data/memory"
    collection_name: str = "transcriber_memory"
    
    # Embedding configuration
    embedding_strategy: str = "sentence_transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Context retrieval settings
    max_context_memories: int = 5
    similarity_threshold: float = 0.7
    context_window_days: int = 30  # Only consider memories from last N days
    
    # Performance optimization
    lazy_initialization: bool = True
    background_processing: bool = True
    embedding_cache_size: int = 1000
    
    # Memory lifecycle
    memory_retention_days: int = 365
    auto_cleanup_enabled: bool = True
    cleanup_interval_hours: int = 24
    
    # Privacy controls
    exclude_sensitive_patterns: list[str] = [
        "password", "secret", "token", "key"
    ]


class Settings(BaseSettings):
    """Main application settings."""
    
    # Audio settings
    audio: AudioConfig = AudioConfig()
    
    # Agent settings
    agent: AgentConfig = AgentConfig()
    
    # Voice processing
    voice: VoiceConfig = VoiceConfig()
    
    # Whisper STT settings
    whisper: WhisperConfig = WhisperConfig()
    
    # Piper TTS settings
    piper: PiperConfig = PiperConfig()
    
    # Session management settings
    session: SessionConfig = SessionConfig()
    
    # Performance monitoring settings
    performance: PerformanceConfig = PerformanceConfig()
    
    # Memory system settings
    memory: MemoryConfig = MemoryConfig()
    
    # Edge TTS voice selection
    edge_voice: str = "en-US-AriaNeural"  # Default to Aria (female, US)
    
    # Application settings
    debug: bool = False
    log_level: str = "INFO"
    data_dir: str = "./data"
    
    class Config:
        env_prefix = "TRANSCRIBER_"
        env_nested_delimiter = "__"


# Global settings instance
settings = Settings()