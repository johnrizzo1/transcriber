# Technical Context: AI Voice Transcriber

## Technology Stack

### Core Runtime
- **Python**: 3.10+ with asyncio for concurrent processing
- **Package Manager**: Poetry for dependency management
- **Environment**: devbox for reproducible development setup
- **Architecture**: Event-driven async pipeline

### AI & Language Models
- **LLM Server**: Ollama (local inference server)
- **Default Model**: llama3.2:3b (tool-calling capable)
- **Alternative Models**: Qwen 2.5, other Ollama-compatible models
- **Context Management**: Conversation memory with SQLite storage

### Speech Processing
- **STT Engine**: faster-whisper (optimized Whisper implementation)
- **TTS Engine**: Edge TTS (primary) / Piper TTS (fallback)
- **VAD**: WebRTC VAD (py-webrtcvad)
- **Audio I/O**: sounddevice (cross-platform audio)

### Development Tools
- **CLI Framework**: Typer with Rich for beautiful terminal UI
- **Configuration**: Pydantic Settings with environment variable support
- **Logging**: Python logging with structured output
- **Testing**: pytest with async support
- **Code Quality**: ruff (linting), black (formatting), mypy (typing)

## Dependencies Overview

### Core Dependencies ([`pyproject.toml`](pyproject.toml))
```toml
# Essential runtime
ollama = "^0.1.7"           # LLM client
pydantic = "^2.5.0"         # Settings and data validation
rich = "^13.0.0"            # Terminal UI
typer = "^0.9.0"            # CLI framework
sounddevice = "^0.4.6"      # Audio I/O

# Speech processing
faster-whisper = "^1.0.0"   # STT engine
edge-tts = "^6.1.0"         # TTS engine
webrtcvad = "^2.0.10"       # Voice activity detection
gtts = "^2.5.0"             # Fallback TTS

# Data and async
aiofiles = "^23.0.0"        # Async file operations
sqlalchemy = "^2.0.0"       # Database ORM
aiosqlite = "^0.19.0"       # Async SQLite
numpy = "^1.24.0"           # Audio processing
```

### Development Dependencies
```toml
pytest = "^7.4.0"           # Testing framework
pytest-asyncio = "^0.21.0"  # Async test support
ruff = "^0.1.0"             # Fast linter
black = "^23.0.0"           # Code formatter
mypy = "^1.5.0"             # Type checking
```

## Configuration System

### Settings Architecture
- **Base Config**: [`transcriber/config.py`](transcriber/config.py)
- **Environment Variables**: `TRANSCRIBER_*` prefix
- **Nested Settings**: Double underscore delimiter (`TRANSCRIBER_AGENT__MODEL`)
- **Validation**: Pydantic models with type checking

### Key Configuration Areas
```python
class Settings(BaseSettings):
    # Agent configuration
    agent: AgentSettings
    
    # Audio processing
    audio: AudioSettings
    whisper: WhisperSettings
    
    # TTS configuration  
    tts: TTSSettings
    
    # Tool system
    tools: ToolSettings
```

### Environment Setup
- **devbox**: Reproducible development environment
- **Poetry**: Virtual environment and dependency management
- **.envrc**: Environment variable loading
- **Ollama**: Must be installed and running separately

## Audio Processing Pipeline

### Audio Configuration
- **Sample Rate**: 16kHz (configurable)
- **Channels**: Mono
- **Chunk Size**: 100ms (1600 samples at 16kHz)
- **Buffer**: Circular buffer for continuous processing
- **Format**: 16-bit PCM

### Processing Chain
1. **Capture**: sounddevice → numpy arrays
2. **VAD**: WebRTC VAD → speech/silence detection
3. **STT**: faster-whisper → text transcription
4. **Agent**: Ollama LLM → response generation
5. **TTS**: Edge TTS → audio synthesis
6. **Output**: sounddevice → speaker playback

## Development Workflow

### Setup Commands
```bash
# Enter development environment
devbox shell

# Install dependencies
poetry install

# Run tests
poetry run pytest

# Code formatting
poetry run black .
poetry run ruff format .

# Type checking
poetry run mypy transcriber/
```

### Running the Application
```bash
# Text-only mode (no speech dependencies needed)
poetry run python chat.py

# Full voice pipeline
poetry run python -m transcriber start

# List audio devices
poetry run python -m transcriber start --list-devices
```

## Performance Characteristics

### Latency Profile (Target: <800ms)
- **Audio capture**: ~20ms (hardware dependent)
- **VAD processing**: ~10ms (WebRTC VAD)
- **STT transcription**: ~150ms (faster-whisper tiny)
- **LLM processing**: ~200ms (llama3.2:3b)
- **TTS synthesis**: ~100ms (Edge TTS)
- **Audio output**: ~20ms (hardware dependent)

### Resource Usage
- **RAM**: ~2-4GB (excluding LLM model)
- **CPU**: Moderate (4-8 cores recommended)
- **Disk**: ~5GB (models and dependencies)
- **Network**: None (after initial setup)

## Integration Requirements

### External Services
- **Ollama Server**: Must be running locally
  - Default: `http://localhost:11434`
  - Models: llama3.2:3b (or compatible)
  - Tool calling support required

### System Requirements
- **Operating System**: macOS (primary), Linux/Windows (future)
- **Python**: 3.10+ with asyncio support
- **Audio**: Working microphone and speakers
- **Memory**: 8GB minimum, 16GB recommended

### Optional Components
- **Piper TTS**: Alternative TTS engine (requires FFmpeg)
- **OpenAI Whisper**: Alternative STT (requires PyTorch)
- **Custom Models**: Any Ollama-compatible LLM

## Deployment Considerations

### Local Development
- All processing runs locally
- No internet required after setup
- Models cached locally
- Configuration via environment variables

### Production Deployment
- Package as standalone application
- Include model downloads in setup
- System service integration
- Resource monitoring and limits

### Security & Privacy
- **No Network Calls**: All processing local
- **Data Retention**: User-controlled conversation storage
- **Model Privacy**: Local inference only
- **Audio Privacy**: No cloud transmission