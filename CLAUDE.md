# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python CLI application called "transcriber" that provides a voice interface for interacting with an AI agent capable of executing tools and actions. The system enables natural voice conversations while maintaining complete privacy through local processing.

## Development Commands

```bash
# Enter development environment (with devbox)
devbox shell

# Install dependencies
poetry install

# Run the application
poetry run python -m transcriber
# or with specific commands:
poetry run python -m transcriber start --list-devices

# Run tests
poetry run pytest

# Run linting and formatting
poetry run ruff check .
poetry run ruff format .
poetry run black .
```

## Project Architecture

### Technology Stack

- **Language**: Python 3.10+
- **Package Manager**: Poetry for dependency management
- **Environment**: Devbox for reproducible development
- **Type**: CLI application with voice interface

### Key Files

- `transcriber/main.py`: Main entry point for the CLI application
- `transcriber/__init__.py`: Package initialization
- `pyproject.toml`: Poetry configuration and dependencies
- `poetry.lock`: Locked dependency versions
- `devbox.json`: Development environment configuration

### Project Goals

1. Create a program to transcribe spoken thoughts
2. Align the transcribed text with the captured audio
3. Allow for synchronized playback of audio and text
4. Enable real-time full-duplex conversations with AI
5. Maintain 100% local processing for privacy

### Technology Decisions

- **Audio Capture**: sounddevice (16kHz sampling) - PyAudio alternative with better cross-platform support
- **Voice Activity Detection**: Silero VAD (via Python bindings) - pending implementation
- **Speech-to-Text**: faster-whisper (streaming mode, tiny/base model) - pending implementation
- **Agent Framework**: LangChain/LangGraph or custom async implementation - pending
- **Language Model**: Ollama Python client with Llama 3.2 3B - tested and working
- **Text-to-Speech**: Piper TTS (Python bindings) - pending implementation
- **Architecture**: Async streaming pipeline targeting <800ms latency

## Development Notes

### Working Features
- ✅ Audio device enumeration and selection
- ✅ Real-time audio capture with level monitoring
- ✅ Ollama integration and LLM communication
- ✅ Async audio streaming architecture
- ✅ Rich CLI interface with beautiful output

### Code Patterns
The project uses Python 3.10+ with async/await for concurrent operations. Key patterns:

```python
# Async streaming example
async for chunk in audio_stream:
    processed = await process_chunk(chunk)
    yield processed
```

### Guidelines
- Use type hints throughout the codebase
- Follow async patterns for all I/O operations
- Implement proper error handling with user-friendly messages
- Keep latency in mind for all processing steps

### Running Commands
```bash
# Quick test of audio system
poetry run python -m transcriber start --device 5

# See all devices
poetry run python -m transcriber start --list-devices
```
