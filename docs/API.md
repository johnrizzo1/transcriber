# API Documentation

Developer reference for the AI Voice Agent's internal APIs and interfaces.

## Table of Contents

1. [Overview](#overview)
2. [Core APIs](#core-apis)
3. [Agent API](#agent-api)
4. [Audio Pipeline API](#audio-pipeline-api)
5. [Tool System API](#tool-system-api)
6. [Session Management API](#session-management-api)
7. [Performance API](#performance-api)
8. [Configuration API](#configuration-api)
9. [Usage Examples](#usage-examples)

## Overview

The AI Voice Agent provides several internal APIs for developers who want to integrate the system into their applications or extend its functionality.

### API Architecture

```python
from transcriber.agent.core import VoiceAgent
from transcriber.tools.registry import ToolRegistry
from transcriber.session.manager import SessionManager
from transcriber.performance.monitor import PerformanceMonitor
from transcriber.config import settings
```

### Basic Integration Example

```python
import asyncio
from transcriber.agent.core import VoiceAgent
from transcriber.config import settings

async def main():
    # Initialize the voice agent
    agent = VoiceAgent(settings)
    await agent.initialize()
    
    try:
        # Process text input
        response = await agent.process_text("What's 2 + 2?")
        print(f"Response: {response}")
        
        # Process voice input (if audio available)
        # audio_data = capture_audio()
        # response = await agent.process_audio(audio_data)
        
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

## Core APIs

### VoiceAgent Class

The main interface for interacting with the AI Voice Agent.

```python
class VoiceAgent:
    def __init__(self, settings: Settings)
    async def initialize(self) -> None
    async def cleanup(self) -> None
    async def process_text(self, text: str) -> str
    async def process_audio(self, audio_data: bytes) -> str
    async def start_voice_session(self) -> None
    async def stop_voice_session(self) -> None
```

#### Methods

**`__init__(settings: Settings)`**
- Initialize the voice agent with configuration settings
- Parameters:
  - `settings`: Configuration object with all system settings

**`async initialize() -> None`**
- Initialize all components (LLM, audio, tools, etc.)
- Must be called before using the agent
- Raises: `InitializationError` if setup fails

**`async cleanup() -> None`**
- Clean up resources and close connections
- Should be called when done using the agent

**`async process_text(text: str) -> str`**
- Process text input and return AI response
- Parameters:
  - `text`: Input text to process
- Returns: AI response as string
- Raises: `ProcessingError` if processing fails

**`async process_audio(audio_data: bytes) -> str`**
- Process audio input and return AI response
- Parameters:
  - `audio_data`: Raw audio bytes (16kHz, mono, 16-bit PCM)
- Returns: AI response as string
- Raises: `AudioProcessingError` if audio processing fails

## Agent API

### LLM Service

Interface for interacting with the language model.

```python
from transcriber.agent.llm import LLMService

class LLMService:
    def __init__(self, settings: AgentSettings)
    async def initialize(self) -> None
    async def generate_response(self, messages: List[Message]) -> str
    async def generate_response_stream(self, messages: List[Message]) -> AsyncIterator[str]
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any
```

#### Usage Example

```python
from transcriber.agent.llm import LLMService
from transcriber.config import settings

async def use_llm():
    llm = LLMService(settings.agent)
    await llm.initialize()
    
    messages = [
        {"role": "user", "content": "What's the weather like?"}
    ]
    
    # Generate complete response
    response = await llm.generate_response(messages)
    print(response)
    
    # Generate streaming response
    async for chunk in llm.generate_response_stream(messages):
        print(chunk, end="", flush=True)
```

### Text Agent

Simplified text-only interface.

```python
from transcriber.agent.text_agent import TextAgent

class TextAgent:
    def __init__(self, settings: Settings)
    async def initialize(self) -> None
    async def process_message(self, message: str) -> str
    async def start_interactive_session(self) -> None
```

## Audio Pipeline API

### Audio Capture

```python
from transcriber.audio.capture import AudioCapture

class AudioCapture:
    def __init__(self, settings: AudioSettings)
    async def start_capture(self) -> None
    async def stop_capture(self) -> None
    async def get_audio_chunk(self) -> bytes
    def list_devices(self) -> List[AudioDevice]
```

### Speech-to-Text

```python
from transcriber.audio.stt import STTProcessor

class STTProcessor:
    def __init__(self, settings: WhisperSettings)
    async def initialize(self) -> None
    async def transcribe(self, audio_data: bytes) -> str
    async def transcribe_stream(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[str]
```

### Text-to-Speech

```python
from transcriber.audio.tts import TTSService

class TTSService:
    def __init__(self, settings: TTSSettings)
    async def initialize(self) -> None
    async def synthesize(self, text: str) -> bytes
    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]
    def list_voices(self) -> List[Voice]
```

### Voice Activity Detection

```python
from transcriber.audio.vad import VADProcessor

class VADProcessor:
    def __init__(self, settings: AudioSettings)
    def process_chunk(self, audio_chunk: bytes) -> bool
    def is_speech(self, audio_data: bytes) -> bool
    def get_speech_segments(self, audio_data: bytes) -> List[Tuple[int, int]]
```

## Tool System API

### Tool Registry

```python
from transcriber.tools.registry import ToolRegistry, get_registry

# Get global registry
registry = get_registry()

# Register a tool
registry.register(my_tool)

# Get tool by name
tool = registry.get("calculator")

# Execute tool
result = await registry.execute_tool("calculator", expression="2+2")

# List all tools
tools = registry.get_all()

# Search tools
matching_tools = registry.search("file")
```

### Creating Custom Tools

```python
from transcriber.tools.base import BaseTool, ToolMetadata, ToolParameter, ToolResult
from transcriber.tools.base import ToolCategory, ToolPermission

class MyCustomTool(BaseTool):
    def _define_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="my_tool",
            description="Description of what my tool does",
            category=ToolCategory.UTILITY,
            version="1.0.0",
            author="Your Name",
            permissions=[ToolPermission.READ],
            examples=[
                "Use my tool to process data",
                "my_tool(input='example')"
            ]
        )
    
    def _define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="input",
                type="str",
                description="Input data to process",
                required=True
            ),
            ToolParameter(
                name="format",
                type="str",
                description="Output format",
                required=False,
                default="json",
                choices=["json", "text", "xml"]
            )
        ]
    
    async def _execute(self, **kwargs) -> Any:
        input_data = kwargs["input"]
        format_type = kwargs.get("format", "json")
        
        # Process the input
        result = self.process_data(input_data, format_type)
        
        return result
    
    def process_data(self, data: str, format_type: str) -> str:
        # Your tool logic here
        return f"Processed {data} in {format_type} format"

# Register the tool
from transcriber.tools.registry import get_registry
registry = get_registry()
registry.register(MyCustomTool())
```

## Session Management API

### Session Manager

```python
from transcriber.session.manager import SessionManager

class SessionManager:
    async def initialize(self) -> None
    async def create_session(self, title: str = None) -> Session
    async def get_session(self, session_id: UUID) -> Optional[Session]
    async def list_sessions(self, limit: int = 20, status: SessionStatus = None) -> List[Session]
    async def save_message(self, session_id: UUID, message: Message) -> None
    async def export_session(self, session_id: UUID, format: str = "json") -> str
    async def replay_session(self, session_id: UUID) -> List[Message]
    async def delete_session(self, session_id: UUID) -> bool
    async def cleanup_old_sessions(self, days: int = 30) -> int
```

#### Usage Example

```python
from transcriber.session.manager import SessionManager
from transcriber.session.models import MessageType

async def manage_sessions():
    manager = SessionManager()
    await manager.initialize()
    
    # Create new session
    session = await manager.create_session("My Conversation")
    
    # Save messages
    await manager.save_message(session.id, Message(
        content="Hello, AI!",
        message_type=MessageType.USER
    ))
    
    await manager.save_message(session.id, Message(
        content="Hello! How can I help you?",
        message_type=MessageType.ASSISTANT
    ))
    
    # Export session
    json_export = await manager.export_session(session.id, "json")
    print(json_export)
    
    # List all sessions
    sessions = await manager.list_sessions()
    for s in sessions:
        print(f"Session: {s.title} ({s.id})")
```

### Session Models

```python
from transcriber.session.models import Session, Message, MessageType, SessionStatus

# Session model
class Session:
    id: UUID
    title: str
    status: SessionStatus
    created_at: datetime
    completed_at: Optional[datetime]
    metadata: SessionMetadata

# Message model
class Message:
    id: UUID
    session_id: UUID
    content: str
    message_type: MessageType
    timestamp: datetime
    metadata: Dict[str, Any]

# Enums
class MessageType(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"

class SessionStatus(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"
```

## Performance API

### Performance Monitor

```python
from transcriber.performance.monitor import PerformanceMonitor

class PerformanceMonitor:
    def __init__(self, settings: PerformanceSettings)
    async def start_monitoring(self) -> None
    async def stop_monitoring(self) -> None
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None) -> None
    def get_metrics(self, name: str = None, duration: int = 3600) -> List[Metric]
    def get_system_stats(self) -> SystemStats
```

#### Usage Example

```python
from transcriber.performance.monitor import PerformanceMonitor
from transcriber.performance.integration import monitor_performance

# Using the monitor directly
monitor = PerformanceMonitor(settings.performance)
await monitor.start_monitoring()

# Record custom metrics
monitor.record_metric("custom_latency", 150.5, {"component": "my_component"})

# Get metrics
metrics = monitor.get_metrics("latency", duration=1800)  # Last 30 minutes

# Using decorator for automatic monitoring
@monitor_performance("my_function")
async def my_function():
    # Function implementation
    await asyncio.sleep(0.1)
    return "result"

result = await my_function()  # Automatically records execution time
```

### Benchmarking

```python
from transcriber.performance.benchmarks import BenchmarkRunner

class BenchmarkRunner:
    def __init__(self, settings: Settings)
    async def run_all_benchmarks(self) -> BenchmarkResults
    async def run_component_benchmark(self, component: str) -> ComponentBenchmark
    async def run_latency_test(self, duration: int = 60) -> LatencyBenchmark
    async def run_stress_test(self, concurrent_requests: int = 10) -> StressBenchmark
```

## Configuration API

### Settings Management

```python
from transcriber.config import Settings, AgentSettings, AudioSettings

# Load settings
settings = Settings()

# Access nested settings
agent_config = settings.agent
audio_config = settings.audio

# Modify settings programmatically
settings.agent.model = "llama3.2:1b"
settings.audio.sample_rate = 22050

# Validate settings
try:
    settings.validate()
except ValidationError as e:
    print(f"Configuration error: {e}")

# Save settings to file
settings.save_to_file("~/.transcriber/config.yaml")

# Load from file
settings = Settings.load_from_file("~/.transcriber/config.yaml")
```

## Usage Examples

### Complete Integration Example

```python
import asyncio
from transcriber.agent.core import VoiceAgent
from transcriber.session.manager import SessionManager
from transcriber.performance.monitor import PerformanceMonitor
from transcriber.tools.registry import get_registry
from transcriber.config import settings

class MyApplication:
    def __init__(self):
        self.agent = VoiceAgent(settings)
        self.session_manager = SessionManager()
        self.performance_monitor = PerformanceMonitor(settings.performance)
        self.tool_registry = get_registry()
    
    async def initialize(self):
        """Initialize all components"""
        await self.agent.initialize()
        await self.session_manager.initialize()
        await self.performance_monitor.start_monitoring()
        
        # Register custom tools
        from my_tools import MyCustomTool
        self.tool_registry.register(MyCustomTool())
    
    async def process_conversation(self, user_input: str) -> str:
        """Process a conversation turn"""
        # Create or get session
        session = await self.session_manager.create_session("API Conversation")
        
        # Save user message
        await self.session_manager.save_message(session.id, Message(
            content=user_input,
            message_type=MessageType.USER
        ))
        
        # Process with agent
        response = await self.agent.process_text(user_input)
        
        # Save assistant response
        await self.session_manager.save_message(session.id, Message(
            content=response,
            message_type=MessageType.ASSISTANT
        ))
        
        return response
    
    async def cleanup(self):
        """Clean up resources"""
        await self.agent.cleanup()
        await self.session_manager.close()
        await self.performance_monitor.stop_monitoring()

# Usage
async def main():
    app = MyApplication()
    await app.initialize()
    
    try:
        # Process some conversations
        response1 = await app.process_conversation("What's 2 + 2?")
        print(f"Response: {response1}")
        
        response2 = await app.process_conversation("List files in current directory")
        print(f"Response: {response2}")
        
    finally:
        await app.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

### Custom Tool Integration

```python
from transcriber.tools.base import BaseTool, ToolMetadata, ToolParameter
from transcriber.tools.registry import get_registry
import requests

class WeatherTool(BaseTool):
    def _define_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="weather",
            description="Get current weather for a location",
            category=ToolCategory.INFORMATION,
            permissions=[ToolPermission.NETWORK]
        )
    
    def _define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="location",
                type="str",
                description="City name or coordinates",
                required=True
            )
        ]
    
    async def _execute(self, location: str) -> str:
        # Implementation would call weather API
        return f"Weather in {location}: Sunny, 72°F"

# Register the tool
registry = get_registry()
registry.register(WeatherTool())

# Now the agent can use the weather tool
agent = VoiceAgent(settings)
await agent.initialize()
response = await agent.process_text("What's the weather in San Francisco?")
```

### Performance Monitoring Integration

```python
from transcriber.performance.integration import monitor_performance
from transcriber.performance.monitor import PerformanceMonitor

# Monitor function performance
@monitor_performance("data_processing")
async def process_large_dataset(data):
    # Simulate processing
    await asyncio.sleep(2)
    return len(data)

# Monitor class methods
class DataProcessor:
    @monitor_performance("DataProcessor.process")
    async def process(self, data):
        return await self.heavy_computation(data)
    
    async def heavy_computation(self, data):
        # Simulate heavy work
        await asyncio.sleep(1)
        return data * 2

# Custom metrics
monitor = PerformanceMonitor(settings.performance)
await monitor.start_monitoring()

# Record custom business metrics
monitor.record_metric("user_requests", 1, {"endpoint": "/api/chat"})
monitor.record_metric("response_quality", 0.95, {"model": "llama3.2:3b"})

# Get performance insights
metrics = monitor.get_metrics("latency", duration=3600)
avg_latency = sum(m.value for m in metrics) / len(metrics)
print(f"Average latency: {avg_latency}ms")
```

## Error Handling

### Common Exceptions

```python
from transcriber.exceptions import (
    TranscriberError,
    InitializationError,
    ProcessingError,
    AudioProcessingError,
    ToolExecutionError,
    SessionError,
    ConfigurationError
)

try:
    agent = VoiceAgent(settings)
    await agent.initialize()
    response = await agent.process_text("Hello")
except InitializationError as e:
    print(f"Failed to initialize: {e}")
except ProcessingError as e:
    print(f"Processing failed: {e}")
except TranscriberError as e:
    print(f"General error: {e}")
```

### Best Practices

1. **Always initialize before use**:
   ```python
   agent = VoiceAgent(settings)
   await agent.initialize()  # Required
   ```

2. **Clean up resources**:
   ```python
   try:
       # Use agent
       pass
   finally:
       await agent.cleanup()
   ```

3. **Handle errors gracefully**:
   ```python
   try:
       response = await agent.process_text(user_input)
   except ProcessingError:
       response = "I'm sorry, I couldn't process that request."
   ```

4. **Monitor performance**:
   ```python
   @monitor_performance("critical_function")
   async def critical_function():
       # Implementation
       pass
   ```

---

This API documentation provides the foundation for integrating and extending the AI Voice Agent. For more examples and advanced usage, see the [User Guide](USER_GUIDE.md) and [Tool Development Guide](TOOL_DEVELOPMENT.md).