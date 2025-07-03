# AI Voice Agent - Transcriber

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A **local, real-time voice interface** for interacting with an AI agent capable of executing tools and actions. The system enables natural voice conversations while maintaining complete privacy through 100% on-device processing.

## 🌟 Features

- 🎤 **Real-time Voice Interaction** - Natural conversations with <800ms latency
- 🤖 **Local AI Processing** - 100% offline using Ollama LLM (Llama 3.2 3B)
- 🛠️ **Extensible Tool System** - 15 built-in tools across 4 categories
- 🔒 **Privacy-First** - No cloud APIs, all processing on-device
- ⚡ **Low-Latency Pipeline** - Streaming audio processing throughout
- 💾 **Session Management** - Persistent conversation storage with SQLite
- 📊 **Performance Monitoring** - Real-time metrics and optimization
- 🎨 **Rich CLI Interface** - Beautiful terminal UI with live updates

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.ai/) installed and running
- Audio input/output devices (microphone and speakers)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/transcriber.git
   cd transcriber
   ```

2. **Install dependencies:**
   ```bash
   # Install Poetry if you haven't already
   curl -sSL https://install.python-poetry.org | python3 -

   # Install project dependencies
   poetry install
   ```

3. **Set up Ollama:**
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.ai/install.sh | sh

   # Download the default model
   ollama pull llama3.2:3b
   ```

### First Run

1. **Test text-only mode** (works immediately):
   ```bash
   poetry run transcriber chat
   ```

2. **List available audio devices:**
   ```bash
   poetry run transcriber start --list-devices
   ```

3. **Start the full voice agent:**
   ```bash
   poetry run transcriber start
   ```

## 🎯 Usage

### Voice Mode

Start the voice agent for natural conversation:

```bash
# Basic voice mode
poetry run transcriber start

# Use specific model
poetry run transcriber start --model llama3.2:1b

# Use specific audio device
poetry run transcriber start --device 2
```

**Voice Controls:**
- Speak naturally to interact with the AI
- Press `q` to interrupt AI speech
- Press `Ctrl+C` to exit

### Text Mode

For development and testing without audio:

```bash
# Interactive text chat
poetry run transcriber chat

# Use different model
poetry run transcriber chat --model llama3.2:1b
```

### Tool Management

Explore and manage the built-in tools:

```bash
# List all available tools
poetry run transcriber list-tools

# Show detailed tool information
poetry run transcriber list-tools --detailed

# Filter by category
poetry run transcriber list-tools --category system

# Search tools
poetry run transcriber list-tools --search "file"
```

### Session Management

Manage conversation history:

```bash
# List recent sessions
poetry run transcriber list-sessions

# Replay the latest session
poetry run transcriber replay

# Replay specific session
poetry run transcriber replay abc12345

# Export session to file
poetry run transcriber export latest --format json
```

### Performance Monitoring

Monitor and optimize performance:

```bash
# Show performance metrics
poetry run transcriber performance

# Run benchmarks
poetry run transcriber benchmark

# Profile performance
poetry run transcriber profile --duration 30
```

## 🛠️ Built-in Tools

The system includes **15 built-in tools** across 4 categories:

### System Tools (5 tools)
- **File Operations** - Read, write, list, delete, copy files
- **Process Management** - List and manage system processes
- **System Information** - Get system stats and information
- **Environment Variables** - Access and manage environment variables
- **System Uptime** - Check system uptime and load

### Utility Tools (4 tools)
- **Calculator** - Basic and advanced mathematical calculations
- **Text Processing** - Analyze, transform, and search text
- **Text Analysis** - Word count, sentiment, readability analysis
- **Text Generator** - Generate text based on patterns

### Information Tools (3 tools)
- **Web Search** - Search the web for information (local processing)
- **Documentation Lookup** - Search documentation and help files
- **Unit Conversion** - Convert between different units

### Productivity Tools (3 tools)
- **Note Taking** - Create and manage notes
- **Task Management** - Create and track TODO items
- **Timer/Reminder** - Set timers and reminders

## 🏗️ Architecture

The system uses a modular, event-driven architecture:

```
Audio Input → VAD → STT → Agent → Tools → TTS → Audio Output
     ↓                     ↓        ↓
Session Recording   Conversation  Performance
                      Memory       Monitoring
```

### Core Components

- **Audio Pipeline** - sounddevice, WebRTC VAD, faster-whisper, Edge TTS
- **AI Agent** - Ollama LLM with tool-calling capabilities
- **Tool System** - Extensible plugin architecture with 15 built-in tools
- **Session Manager** - SQLite-based conversation persistence
- **Performance Monitor** - Real-time metrics and optimization

## 📊 Performance

### Current Metrics
- **End-to-end Latency**: ~800ms (target achieved)
- **Audio Quality**: 16kHz clean processing
- **Memory Usage**: ~2-4GB (excluding LLM models)
- **Tool Execution**: >90% success rate
- **Speech Accuracy**: >95% in quiet environments

### Latency Breakdown
- Audio capture: ~20ms
- VAD processing: ~10ms
- STT transcription: ~150ms
- LLM processing: ~200ms
- Tool execution: 100-500ms (varies)
- TTS synthesis: ~100ms
- Audio output: ~20ms

## 🔧 Configuration

### Environment Variables

```bash
# Agent configuration
export TRANSCRIBER_AGENT__MODEL="llama3.2:3b"
export TRANSCRIBER_AGENT__TEMPERATURE=0.7
export TRANSCRIBER_AGENT__BASE_URL="http://localhost:11434"

# Audio configuration
export TRANSCRIBER_AUDIO__SAMPLE_RATE=16000
export TRANSCRIBER_AUDIO__INPUT_DEVICE=0

# Speech processing
export TRANSCRIBER_WHISPER__MODEL="tiny"
export TRANSCRIBER_TTS__ENGINE="edge-tts"

# Performance
export TRANSCRIBER_PERFORMANCE__ENABLE_MONITORING=true
export TRANSCRIBER_PERFORMANCE__METRICS_INTERVAL=5
```

### Configuration File

Create `~/.transcriber/config.yaml` for persistent settings:

```yaml
agent:
  model: "llama3.2:3b"
  temperature: 0.7
  max_tokens: 2048

audio:
  sample_rate: 16000
  chunk_size: 1600
  input_device: null  # Auto-detect

whisper:
  model: "tiny"
  language: "en"

tts:
  engine: "edge-tts"
  voice: "en-US-AriaNeural"
  rate: "+0%"

performance:
  enable_monitoring: true
  metrics_interval: 5
  enable_profiling: false
```

## 🧪 Development

### Setup Development Environment

```bash
# Clone and enter directory
git clone https://github.com/yourusername/transcriber.git
cd transcriber

# Install with development dependencies
poetry install --with dev

# Install pre-commit hooks
poetry run pre-commit install
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=transcriber

# Run specific test categories
poetry run pytest tests/unit/
poetry run pytest tests/integration/
poetry run pytest tests/performance/
```

### Code Quality

```bash
# Format code
poetry run black .
poetry run ruff format .

# Lint code
poetry run ruff check .

# Type checking
poetry run mypy transcriber/
```

### Creating Custom Tools

See [`docs/TOOL_DEVELOPMENT.md`](docs/TOOL_DEVELOPMENT.md) for detailed instructions on creating custom tools.

## 📚 Documentation

- **[User Guide](docs/USER_GUIDE.md)** - Complete usage instructions
- **[Installation Guide](docs/INSTALLATION_GUIDE.md)** - Detailed setup instructions
- **[CLI Reference](docs/CLI_REFERENCE.md)** - All commands with examples
- **[Tool Documentation](docs/TOOLS.md)** - All 15 tools with usage examples
- **[Architecture](docs/ARCHITECTURE.md)** - System design and components
- **[API Documentation](docs/API.md)** - Internal APIs and interfaces
- **[Performance Guide](docs/PERFORMANCE.md)** - Performance monitoring and optimization
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[Contributing](docs/CONTRIBUTING.md)** - Development workflow and standards

## 🐛 Troubleshooting

### Common Issues

**Audio device not found:**
```bash
# List available devices
poetry run transcriber start --list-devices

# Use specific device
poetry run transcriber start --device 2
```

**Ollama connection failed:**
```bash
# Check if Ollama is running
ollama list

# Start Ollama service
ollama serve

# Pull required model
ollama pull llama3.2:3b
```

**Speech recognition not working:**
```bash
# Test with text mode first
poetry run transcriber chat

# Check microphone permissions
# Ensure microphone is not muted
```

For more troubleshooting, see [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).

## 🤝 Contributing

We welcome contributions! Please see [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) for:

- Development setup
- Code style guidelines
- Testing requirements
- Pull request process
- Tool development guide

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for local LLM inference
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) for efficient speech recognition
- [Edge TTS](https://github.com/rany2/edge-tts) for high-quality text-to-speech
- [Rich](https://github.com/Textualize/rich) for beautiful terminal interfaces
- [Typer](https://github.com/tiangolo/typer) for CLI framework

## 🔗 Links

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/transcriber/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/transcriber/discussions)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

---

**Ready to start your voice-powered AI journey?** 🚀

```bash
poetry run transcriber start