# System Patterns: AI Voice Transcriber

## Architecture Overview

### Core Pipeline Pattern
```
Audio Input → VAD → STT → Agent → TTS → Audio Output
     ↓                      ↓
Session Recording    Conversation Memory
```

**Key Characteristics:**
- **Async Streaming**: All components use asyncio for non-blocking processing
- **Event-Driven**: Components communicate via async queues and events
- **Modular Design**: Each component is independently replaceable
- **Graceful Degradation**: Text-only fallback when speech components unavailable

### Component Architecture

#### 1. Audio Pipeline (`transcriber/audio/`)
- **Capture**: [`capture.py`](transcriber/audio/capture.py) - sounddevice-based audio input
- **VAD**: [`vad.py`](transcriber/audio/vad.py) - WebRTC VAD with VADProcessor class
- **STT**: [`stt.py`](transcriber/audio/stt.py) - faster-whisper integration
- **TTS**: [`tts.py`](transcriber/audio/tts.py) - Piper TTS / Edge TTS
- **Output**: [`output.py`](transcriber/audio/output.py) - Audio playback with interruption

#### 2. Agent System (`transcriber/agent/`)
- **LLM**: [`llm.py`](transcriber/agent/llm.py) - Ollama client with streaming
- **Core**: [`core.py`](transcriber/agent/core.py) - Main agent orchestrator
- **Text Agent**: [`text_agent.py`](transcriber/agent/text_agent.py) - Fallback mode

#### 3. Tool System (`transcriber/tools/`)
- **Base**: [`base.py`](transcriber/tools/base.py) - Tool interface definition
- **Registry**: [`registry.py`](transcriber/tools/registry.py) - Tool discovery and management
- **Built-ins**: [`builtin/`](transcriber/tools/builtin/) - System, utility, productivity tools

## Key Design Patterns

### 1. Async Queue Pattern
```python
# Used throughout for component communication
audio_queue = asyncio.Queue()
text_queue = asyncio.Queue()
response_queue = asyncio.Queue()
```

### 2. Configuration Pattern
```python
# Centralized settings with environment override
# transcriber/config.py
class Settings(BaseSettings):
    class Config:
        env_prefix = "TRANSCRIBER_"
        env_nested_delimiter = "__"
```

### 3. Graceful Fallback Pattern
```python
# Components check availability and provide alternatives
try:
    from .whisper_stt import WhisperSTT
    stt_class = WhisperSTT
except ImportError:
    from .mock_stt import MockSTT
    stt_class = MockSTT
```

### 4. Tool Interface Pattern
```python
class Tool(ABC):
    @abstractmethod
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        pass
```

## Critical Implementation Paths

### 1. Voice Pipeline Flow
1. **Audio Capture** → Continuous 16kHz mono audio chunks
2. **VAD Processing** → Speech/silence detection with WebRTC
3. **STT Processing** → faster-whisper transcription
4. **Agent Processing** → Ollama LLM with conversation context
5. **TTS Generation** → Edge TTS or Piper TTS synthesis
6. **Audio Output** → Playback with interrupt capability

### 2. Tool Execution Flow
1. **Intent Recognition** → LLM identifies tool requirements
2. **Parameter Extraction** → Parse tool parameters from context
3. **Tool Selection** → Registry lookup and validation
4. **Execution** → Async tool execution with timeout
5. **Result Processing** → Format output for user consumption

### 3. Session Management
- **Conversation Memory** → SQLite storage with context windows
- **Audio Recording** → Optional session recording with timestamps
- **State Persistence** → Agent state across interruptions

## Performance Optimizations

### Latency Targets (Achieved ~800ms total)
- Audio capture: 20ms
- VAD processing: 10ms  
- STT: 150ms
- LLM processing: 200ms
- TTS generation: 100ms
- Audio output: 20ms
- Tool execution: Variable (100-500ms)

### Memory Management
- **Model Caching**: Keep frequently used models loaded
- **Buffer Management**: Circular buffers for audio processing
- **Context Windows**: Limit conversation history size
- **Resource Cleanup**: Proper async resource management

## Error Handling Patterns

### 1. Component Isolation
Each component handles its own errors without crashing the pipeline:
```python
try:
    result = await component.process(data)
except ComponentError as e:
    logger.error(f"Component failed: {e}")
    result = fallback_result
```

### 2. Graceful Degradation
- STT failure → Text input mode
- TTS failure → Text output mode  
- Tool failure → Error message with context
- Audio failure → Continue with available components

### 3. Recovery Mechanisms
- Automatic reconnection for network components
- Model reloading on corruption
- Queue clearing on overflow
- State reset on critical errors

## Integration Points

### External Dependencies
- **Ollama**: Local LLM server (must be running)
- **Audio System**: Platform audio drivers
- **Models**: Whisper, Piper TTS voice models
- **Python Packages**: See [`pyproject.toml`](pyproject.toml)

### Configuration Files
- **Main Config**: [`transcriber/config.py`](transcriber/config.py)
- **Dependencies**: [`pyproject.toml`](pyproject.toml)
- **Environment**: [`.envrc`](.envrc) for devbox
- **Development**: [`devbox.json`](devbox.json)