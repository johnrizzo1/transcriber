# AI Voice Agent - User Guide

This comprehensive guide covers everything you need to know to use the AI Voice Agent effectively, from basic setup to advanced features.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Voice Interaction](#voice-interaction)
3. [Text Mode](#text-mode)
4. [Using Tools](#using-tools)
5. [Session Management](#session-management)
6. [Performance Monitoring](#performance-monitoring)
7. [Configuration](#configuration)
8. [Advanced Usage](#advanced-usage)
9. [Tips and Best Practices](#tips-and-best-practices)

## Getting Started

### First Time Setup

1. **Verify Installation**
   ```bash
   # Check if transcriber is installed
   poetry run transcriber --help
   
   # Should show available commands
   ```

2. **Test Ollama Connection**
   ```bash
   # Check if Ollama is running
   ollama list
   
   # If not running, start it
   ollama serve
   
   # Pull the default model if needed
   ollama pull llama3.2:3b
   ```

3. **Test Audio Devices**
   ```bash
   # List available audio devices
   poetry run transcriber start --list-devices
   
   # Look for your microphone and speakers
   ```

4. **First Test Run**
   ```bash
   # Start with text mode (no audio needed)
   poetry run transcriber chat
   
   # Try a simple question
   > Hello, what can you help me with?
   ```

### Understanding the Interface

When you start the voice agent, you'll see a rich terminal interface with several sections:

```
┌─ Transcriber AI Voice Agent ─┐
│ Status: Listening             │
│ Model: llama3.2:3b           │
│ Latency: ~800ms              │
└───────────────────────────────┘

┌─ Audio Levels ─┐
│ ████████░░ 80% │
└────────────────┘

┌─ Conversation ─┐
│ You: Hello     │
│ AI: Hi there!  │
└────────────────┘

Press 'q' to interrupt, Ctrl+C to exit
```

## Voice Interaction

### Starting Voice Mode

```bash
# Basic voice mode with default settings
poetry run transcriber start

# Use a different model
poetry run transcriber start --model llama3.2:1b

# Use specific audio device
poetry run transcriber start --device 2
```

### Voice Controls

- **Speaking**: Just talk naturally - the system detects when you start and stop speaking
- **Interrupting**: Press `q` to stop the AI mid-speech and speak again
- **Exiting**: Press `Ctrl+C` to exit the application
- **Volume**: The audio level indicator shows your microphone input

### Best Practices for Voice Interaction

1. **Speak Clearly**: Use normal conversational tone and pace
2. **Wait for Silence**: Let the AI finish before speaking again (unless interrupting)
3. **Natural Language**: Use natural sentences, not commands
4. **Quiet Environment**: Background noise can affect recognition accuracy

### Example Voice Conversations

**Basic Information:**
```
You: "What's the weather like today?"
AI: "I don't have access to current weather data, but I can help you with many other tasks. Would you like me to show you what tools I have available?"
```

**File Operations:**
```
You: "Can you list the files in my current directory?"
AI: "I'll list the files in your current directory for you."
[Tool executes: file_list]
AI: "Here are the files I found: README.md, pyproject.toml, transcriber/, docs/, tests/..."
```

**Calculations:**
```
You: "What's 15% of 250?"
AI: "Let me calculate that for you."
[Tool executes: calculator]
AI: "15% of 250 is 37.5"
```

## Text Mode

Text mode is perfect for development, testing, or when audio isn't available.

### Starting Text Mode

```bash
# Interactive text chat
poetry run transcriber chat

# Use different model
poetry run transcriber chat --model llama3.2:1b
```

### Text Mode Features

- **Immediate Response**: No audio processing delays
- **Full Tool Access**: All tools work the same as in voice mode
- **Development Friendly**: Perfect for testing and debugging
- **Copy/Paste**: Easy to copy responses or paste complex inputs

### Example Text Session

```
$ poetry run transcriber chat
Starting text-only chat mode...

You: Hello! Can you help me analyze a text file?
AI: Hello! I'd be happy to help you analyze a text file. I have several text processing tools available that can:

- Read and analyze text files
- Count words, characters, and lines
- Search for specific patterns
- Transform text (uppercase, lowercase, etc.)
- Generate text summaries

What specific analysis would you like me to perform? Just let me know the file path and what you'd like to analyze.

You: Can you read the README.md file and tell me how many words it contains?
AI: I'll read the README.md file and analyze it for you.

[Using file_read tool to read README.md]
[Using text_analysis tool to count words]

The README.md file contains 1,247 words. Here's a quick breakdown:
- Words: 1,247
- Characters: 7,832
- Lines: 156
- Paragraphs: 42

The file appears to be a comprehensive project README with sections covering installation, usage, features, and documentation.

You: exit
Goodbye!
```

## Using Tools

The AI Voice Agent comes with 15 built-in tools across 4 categories. Tools are automatically invoked when the AI determines they're needed to fulfill your request.

### Discovering Available Tools

```bash
# List all tools
poetry run transcriber list-tools

# Show detailed information
poetry run transcriber list-tools --detailed

# Filter by category
poetry run transcriber list-tools --category system
poetry run transcriber list-tools --category utility
poetry run transcriber list-tools --category information
poetry run transcriber list-tools --category productivity

# Search for specific tools
poetry run transcriber list-tools --search "file"
poetry run transcriber list-tools --search "calculator"
```

### Tool Categories and Usage

#### System Tools

**File Operations:**
- "Read the contents of config.py"
- "Create a new file called notes.txt with some sample content"
- "List all Python files in the current directory"
- "Delete the temporary file temp.log"
- "Copy README.md to README_backup.md"

**Process Management:**
- "Show me all running Python processes"
- "What processes are using the most CPU?"
- "Kill the process with PID 1234"

**System Information:**
- "What's my system information?"
- "How much memory is available?"
- "What's the current CPU usage?"
- "Show me the system uptime"

**Environment Variables:**
- "What's the value of PATH?"
- "Set the environment variable DEBUG to true"
- "Show all environment variables containing 'PYTHON'"

#### Utility Tools

**Calculator:**
- "What's 15% of 250?"
- "Calculate the square root of 144"
- "What's 2 to the power of 8?"
- "Solve the equation: (5 + 3) * 2 - 4"

**Text Processing:**
- "Count the words in this text: 'Hello world, this is a test'"
- "Convert this text to uppercase: 'hello world'"
- "Search for the word 'python' in the README file"
- "Generate a random sentence about technology"

#### Information Tools

**Web Search:**
- "Search for information about Python asyncio"
- "Find recent news about artificial intelligence"
- "Look up the documentation for FastAPI"

**Documentation Lookup:**
- "Find help for the 'ls' command"
- "Show me Python documentation for the 'asyncio' module"
- "Look up usage examples for 'git commit'"

**Unit Conversion:**
- "Convert 100 fahrenheit to celsius"
- "How many kilometers is 50 miles?"
- "Convert 2.5 hours to minutes"

#### Productivity Tools

**Note Taking:**
- "Create a note titled 'Meeting Notes' with today's agenda"
- "Add a note about the new project requirements"
- "Show me all my notes from this week"

**Task Management:**
- "Add a task to review the documentation"
- "Mark the 'setup environment' task as completed"
- "Show me all pending tasks"

**Timer/Reminder:**
- "Set a timer for 25 minutes for a Pomodoro session"
- "Remind me in 1 hour to check the server logs"
- "Show me all active timers"

### Tool Execution Flow

When you make a request that requires tools:

1. **Intent Recognition**: The AI analyzes your request
2. **Tool Selection**: Appropriate tools are identified
3. **Parameter Extraction**: Required parameters are extracted from context
4. **Execution**: Tools run with proper error handling
5. **Result Integration**: Tool outputs are incorporated into the response
6. **Natural Response**: The AI provides a natural language response

### Tool Permissions

Tools have different permission levels:

- **READ**: Can read files and data
- **WRITE**: Can modify files and data
- **EXECUTE**: Can run programs and commands
- **NETWORK**: Can access network resources
- **SYSTEM**: Can modify system settings

The system will inform you when tools with higher permissions are used.

## Session Management

All conversations are automatically saved and can be replayed, exported, or analyzed.

### Viewing Sessions

```bash
# List recent sessions
poetry run transcriber list-sessions

# Show more sessions
poetry run transcriber list-sessions --limit 50

# Filter by status
poetry run transcriber list-sessions --status active
poetry run transcriber list-sessions --status completed

# Search sessions
poetry run transcriber list-sessions --search "file operations"

# Sort sessions
poetry run transcriber list-sessions --sort created_at --reverse
```

### Session Information

Each session includes:
- **Unique ID**: Short identifier (e.g., `abc12345`)
- **Title**: Auto-generated based on conversation content
- **Status**: `active`, `completed`, or `error`
- **Created Time**: When the session started
- **Duration**: How long the conversation lasted
- **Message Count**: Number of messages exchanged

### Replaying Sessions

```bash
# Replay the most recent session
poetry run transcriber replay

# Replay specific session (use short ID)
poetry run transcriber replay abc12345

# Replay with different formats
poetry run transcriber replay --format plain
poetry run transcriber replay --format rich
poetry run transcriber replay --format json

# Hide timestamps
poetry run transcriber replay --no-timestamps
```

### Exporting Sessions

```bash
# Export latest session as JSON
poetry run transcriber export

# Export specific session
poetry run transcriber export abc12345

# Different formats
poetry run transcriber export --format json
poetry run transcriber export --format txt
poetry run transcriber export --format markdown

# Save to specific file
poetry run transcriber export --output my_conversation.json
```

### Session Cleanup

```bash
# Clean up old sessions (interactive)
poetry run transcriber cleanup

# Clean up sessions older than 30 days
poetry run transcriber cleanup --days 30

# Clean up completed sessions only
poetry run transcriber cleanup --status completed

# Dry run (show what would be deleted)
poetry run transcriber cleanup --dry-run
```

## Performance Monitoring

The system includes comprehensive performance monitoring to help you optimize your experience.

### Real-time Performance

```bash
# Show live performance metrics
poetry run transcriber performance

# Show performance for specific duration
poetry run transcriber performance --duration 60

# Include detailed metrics
poetry run transcriber performance --detailed
```

### Performance Metrics

The system tracks:

- **Latency**: End-to-end response times
- **Component Times**: Individual pipeline stage timings
- **Memory Usage**: RAM consumption by components
- **CPU Usage**: Processing load
- **Tool Performance**: Tool execution times
- **Audio Quality**: Input/output audio metrics

### Benchmarking

```bash
# Run standard benchmarks
poetry run transcriber benchmark

# Run specific benchmark categories
poetry run transcriber benchmark --category audio
poetry run transcriber benchmark --category llm
poetry run transcriber benchmark --category tools

# Save benchmark results
poetry run transcriber benchmark --output benchmark_results.json
```

### Profiling

```bash
# Profile performance for 30 seconds
poetry run transcriber profile --duration 30

# Profile specific components
poetry run transcriber profile --component audio
poetry run transcriber profile --component agent

# Save profiling results
poetry run transcriber profile --output profile_results.json
```

## Configuration

### Environment Variables

Set these in your shell or `.env` file:

```bash
# Agent Configuration
export TRANSCRIBER_AGENT__MODEL="llama3.2:3b"
export TRANSCRIBER_AGENT__TEMPERATURE=0.7
export TRANSCRIBER_AGENT__MAX_TOKENS=2048
export TRANSCRIBER_AGENT__BASE_URL="http://localhost:11434"

# Audio Configuration
export TRANSCRIBER_AUDIO__SAMPLE_RATE=16000
export TRANSCRIBER_AUDIO__CHUNK_SIZE=1600
export TRANSCRIBER_AUDIO__INPUT_DEVICE=0
export TRANSCRIBER_AUDIO__OUTPUT_DEVICE=0

# Speech Processing
export TRANSCRIBER_WHISPER__MODEL="tiny"
export TRANSCRIBER_WHISPER__LANGUAGE="en"
export TRANSCRIBER_TTS__ENGINE="edge-tts"
export TRANSCRIBER_TTS__VOICE="en-US-AriaNeural"

# Performance
export TRANSCRIBER_PERFORMANCE__ENABLE_MONITORING=true
export TRANSCRIBER_PERFORMANCE__METRICS_INTERVAL=5

# Session Management
export TRANSCRIBER_SESSION__AUTO_SAVE=true
export TRANSCRIBER_SESSION__RETENTION_DAYS=30
```

### Configuration File

Create `~/.transcriber/config.yaml`:

```yaml
agent:
  model: "llama3.2:3b"
  temperature: 0.7
  max_tokens: 2048
  base_url: "http://localhost:11434"
  timeout: 30

audio:
  sample_rate: 16000
  chunk_size: 1600
  input_device: null  # Auto-detect
  output_device: null  # Auto-detect
  vad_threshold: 0.5

whisper:
  model: "tiny"  # tiny, base, small, medium, large
  language: "en"
  compute_type: "float16"

tts:
  engine: "edge-tts"  # edge-tts, gtts
  voice: "en-US-AriaNeural"
  rate: "+0%"
  volume: "+0%"

performance:
  enable_monitoring: true
  metrics_interval: 5
  enable_profiling: false
  max_history: 1000

session:
  auto_save: true
  retention_days: 30
  export_format: "json"
  backup_enabled: true

tools:
  enabled_categories:
    - system
    - utility
    - information
    - productivity
  permission_prompts: true
  timeout: 30
```

### Interactive Configuration

```bash
# Launch interactive configuration wizard
poetry run transcriber configure

# Configure specific sections
poetry run transcriber configure --section audio
poetry run transcriber configure --section agent
poetry run transcriber configure --section performance
```

## Advanced Usage

### Custom Models

You can use different Ollama models:

```bash
# List available models
ollama list

# Pull a different model
ollama pull llama3.2:1b
ollama pull qwen2.5:7b
ollama pull codellama:7b

# Use the model
poetry run transcriber start --model llama3.2:1b
```

### Multiple Audio Devices

```bash
# List devices with details
poetry run transcriber start --list-devices

# Use specific input and output devices
export TRANSCRIBER_AUDIO__INPUT_DEVICE=2
export TRANSCRIBER_AUDIO__OUTPUT_DEVICE=3
poetry run transcriber start
```

### Custom Tool Development

See [`docs/TOOL_DEVELOPMENT.md`](TOOL_DEVELOPMENT.md) for creating custom tools.

### Integration with Other Systems

The voice agent can be integrated with other systems:

```python
# Python API usage
from transcriber.agent.core import VoiceAgent
from transcriber.config import settings

async def main():
    agent = VoiceAgent(settings)
    await agent.initialize()
    
    # Process text input
    response = await agent.process_text("What's 2 + 2?")
    print(response)
    
    await agent.cleanup()
```

## Tips and Best Practices

### Voice Interaction Tips

1. **Speak Naturally**: Use conversational language, not commands
2. **Be Specific**: "Read the config.py file" vs "read file"
3. **Wait for Completion**: Let the AI finish before speaking again
4. **Use Interruption Wisely**: Press 'q' to interrupt when needed
5. **Quiet Environment**: Minimize background noise for better recognition

### Performance Optimization

1. **Choose Right Model**: Use smaller models (1b, 3b) for faster responses
2. **Monitor Resources**: Use performance monitoring to identify bottlenecks
3. **Optimize Audio**: Use appropriate sample rates and chunk sizes
4. **Clean Sessions**: Regularly clean up old sessions to save space

### Tool Usage Tips

1. **Be Descriptive**: "Calculate the square root of 144" vs "sqrt 144"
2. **Provide Context**: Include file paths, specific parameters
3. **Check Permissions**: Understand what permissions tools require
4. **Combine Operations**: "Read the file and count the words"

### Session Management

1. **Regular Cleanup**: Clean old sessions to maintain performance
2. **Export Important**: Export important conversations before cleanup
3. **Use Search**: Search sessions by content to find specific conversations
4. **Monitor Storage**: Keep an eye on database size

### Troubleshooting

1. **Start Simple**: Use text mode to test basic functionality
2. **Check Logs**: Look at console output for error messages
3. **Test Components**: Test audio, Ollama, and tools separately
4. **Use Monitoring**: Performance metrics can reveal issues

For more troubleshooting help, see [`docs/TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

---

This guide covers the essential usage patterns for the AI Voice Agent. For more specific information, check the other documentation files in the [`docs/`](.) directory.