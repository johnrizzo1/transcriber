# Project Brief: AI Voice Transcriber Agent

## Core Mission
Build a **local, real-time voice interface** for interacting with an AI agent capable of executing tools and actions. The system enables natural voice conversations while maintaining complete privacy through 100% on-device processing.

## Key Requirements

### Primary Goals
- **Real-time Voice Interaction**: Natural conversations with <800ms latency
- **Local AI Processing**: 100% offline using Ollama LLM (Llama 3.2 3B)
- **Extensible Tool System**: Plugin architecture for various capabilities
- **Privacy-First**: No cloud APIs, all processing on-device
- **Low-Latency Pipeline**: Streaming audio processing throughout

### Technical Foundation
- **Runtime**: Python 3.10+ with asyncio architecture
- **STT**: faster-whisper for speech recognition
- **TTS**: Piper TTS / Edge TTS for speech synthesis
- **LLM**: Ollama with tool-calling models
- **Audio**: sounddevice for cross-platform audio I/O
- **VAD**: WebRTC VAD for speech detection

### Success Metrics
- End-to-end latency: <800ms (target achieved: ~800ms)
- Audio quality: 16kHz sample rate minimum
- Memory usage: <4GB RAM (excluding LLM models)
- Speech accuracy: >95% in quiet environments
- Tool execution: >90% success rate

## Current Status
**Phase 1-6 COMPLETED**: Core voice agent pipeline is fully functional with:
- Complete audio capture → VAD → STT → Agent → TTS → output pipeline
- Async streaming architecture with Rich terminal UI
- Text-only fallback mode for development
- Ollama LLM integration with conversation memory
- Foundation ready for tool system integration

**Next Priority**: Phase 5 (Built-in Tools) and Phase 8 (Performance Optimization)