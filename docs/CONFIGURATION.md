# Configuration Guide

Complete guide to configuring the AI Voice Agent for optimal performance and customization.

## Table of Contents

1. [Configuration Overview](#configuration-overview)
2. [Configuration Methods](#configuration-methods)
3. [Agent Settings](#agent-settings)
4. [Audio Configuration](#audio-configuration)
5. [Speech Processing](#speech-processing)
6. [Performance Settings](#performance-settings)
7. [Session Management](#session-management)
8. [Tool Configuration](#tool-configuration)
9. [Environment Variables](#environment-variables)
10. [Advanced Configuration](#advanced-configuration)

## Configuration Overview

The AI Voice Agent uses a hierarchical configuration system with multiple sources:

1. **Default values** (built into the application)
2. **Configuration file** (`~/.transcriber/config.yaml`)
3. **Environment variables** (prefix: `TRANSCRIBER_`)
4. **Command-line arguments** (highest priority)

### Configuration Priority

```
Command Line Args > Environment Variables > Config File > Defaults
```

### Quick Configuration

```bash
# Interactive configuration wizard
poetry run transcriber configure

# Configure specific sections
poetry run transcriber configure --section audio
poetry run transcriber configure --section agent
```

## Configuration Methods

### 1. Configuration File

Create `~/.transcriber/config.yaml`:

```yaml
# AI Voice Agent Configuration
# Complete configuration with all available options

agent:
  model: "llama3.2:3b"           # Ollama model name
  temperature: 0.7               # Response creativity (0.0-1.0)
  max_tokens: 2048              # Maximum response length
  base_url: "http://localhost:11434"  # Ollama server URL
  timeout: 30                   # Request timeout in seconds
  system_prompt: |              # Custom system prompt
    You are a helpful AI assistant with access to various tools.
    Be concise but thorough in your responses.
  context_window: 4096          # Context window size
  stream_responses: true        # Enable streaming responses

audio:
  sample_rate: 16000            # Audio sample rate (Hz)
  chunk_size: 1600              # Audio chunk size (samples)
  channels: 1                   # Audio channels (1=mono, 2=stereo)
  input_device: null            # Input device index (null=auto)
  output_device: null           # Output device index (null=auto)
  buffer_size: 8192             # Audio buffer size
  vad_threshold: 0.5            # Voice activity detection threshold
  echo_cancellation: true       # Enable echo cancellation
  noise_suppression: true       # Enable noise suppression
  auto_gain_control: true       # Enable automatic gain control

whisper:
  model: "tiny"                 # Whisper model (tiny, base, small, medium, large)
  language: "en"                # Language code (en, es, fr, etc.)
  compute_type: "float16"       # Compute precision (float16, float32, int8)
  device: "cpu"                 # Device (cpu, cuda, auto)
  beam_size: 5                  # Beam search size
  best_of: 5                    # Number of candidates
  temperature: 0.0              # Sampling temperature
  condition_on_previous_text: true  # Use previous text as context
  no_speech_threshold: 0.6      # No speech detection threshold
  logprob_threshold: -1.0       # Log probability threshold
  compression_ratio_threshold: 2.4  # Compression ratio threshold

tts:
  engine: "edge-tts"            # TTS engine (edge-tts, gtts, piper)
  voice: "en-US-AriaNeural"     # Voice name
  rate: "+0%"                   # Speech rate (-50% to +100%)
  volume: "+0%"                 # Speech volume (-50% to +50%)
  pitch: "+0Hz"                 # Voice pitch (-200Hz to +200Hz)
  quality: "high"               # Audio quality (low, medium, high)
  cache_enabled: true           # Enable TTS caching
  cache_size: 100               # Maximum cached items

performance:
  enable_monitoring: true       # Enable performance monitoring
  metrics_interval: 5           # Metrics collection interval (seconds)
  enable_profiling: false       # Enable detailed profiling
  max_history: 1000            # Maximum metrics history
  cpu_threshold: 80            # CPU usage warning threshold (%)
  memory_threshold: 85         # Memory usage warning threshold (%)
  latency_threshold: 1000      # Latency warning threshold (ms)
  auto_optimization: true      # Enable automatic optimization

session:
  auto_save: true              # Automatically save conversations
  retention_days: 30           # Keep sessions for N days
  export_format: "json"        # Default export format
  backup_enabled: true         # Enable session backups
  backup_interval: 24          # Backup interval (hours)
  max_sessions: 1000          # Maximum stored sessions
  compression: true            # Compress stored sessions

tools:
  enabled_categories:          # Enabled tool categories
    - system
    - utility
    - information
    - productivity
  permission_prompts: true     # Prompt for dangerous operations
  timeout: 30                 # Tool execution timeout (seconds)
  max_concurrent: 3           # Maximum concurrent tool executions
  sandbox_enabled: true       # Enable tool sandboxing
  allowed_paths:              # Allowed file system paths
    - "."
    - "~/Documents"
    - "~/Downloads"
  blocked_commands:           # Blocked system commands
    - "rm -rf"
    - "sudo"
    - "chmod 777"

logging:
  level: "INFO"               # Log level (DEBUG, INFO, WARNING, ERROR)
  file: "~/.transcriber/logs/transcriber.log"  # Log file path
  max_size: "10MB"           # Maximum log file size
  backup_count: 5            # Number of backup log files
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  console_output: true       # Enable console logging

ui:
  theme: "dark"              # UI theme (dark, light, auto)
  show_timestamps: true      # Show message timestamps
  show_latency: true         # Show response latency
  show_audio_levels: true    # Show audio level indicators
  animation_speed: "normal"  # Animation speed (slow, normal, fast)
  compact_mode: false        # Enable compact display mode
```

### 2. Environment Variables

Set environment variables with the `TRANSCRIBER_` prefix:

```bash
# Agent configuration
export TRANSCRIBER_AGENT__MODEL="llama3.2:3b"
export TRANSCRIBER_AGENT__TEMPERATURE=0.7
export TRANSCRIBER_AGENT__MAX_TOKENS=2048

# Audio configuration
export TRANSCRIBER_AUDIO__SAMPLE_RATE=16000
export TRANSCRIBER_AUDIO__INPUT_DEVICE=0

# Speech processing
export TRANSCRIBER_WHISPER__MODEL="base"
export TRANSCRIBER_TTS__ENGINE="edge-tts"

# Performance
export TRANSCRIBER_PERFORMANCE__ENABLE_MONITORING=true
```

### 3. Command Line Arguments

Override settings via command line:

```bash
# Use different model
poetry run transcriber start --model llama3.2:1b

# Use specific audio device
poetry run transcriber start --device 2

# Chat with different model
poetry run transcriber chat --model qwen2.5:3b
```

## Agent Settings

### Model Configuration

```yaml
agent:
  model: "llama3.2:3b"          # Available models:
                                # - llama3.2:1b (fastest)
                                # - llama3.2:3b (balanced)
                                # - llama3.2:7b (best quality)
                                # - qwen2.5:3b (alternative)
  
  temperature: 0.7              # Creativity level:
                                # - 0.0: Deterministic, factual
                                # - 0.5: Balanced
                                # - 1.0: Creative, varied
  
  max_tokens: 2048              # Response length limit
  context_window: 4096          # Conversation memory size
```

### Connection Settings

```yaml
agent:
  base_url: "http://localhost:11434"  # Ollama server URL
  timeout: 30                         # Request timeout (seconds)
  retry_attempts: 3                   # Number of retry attempts
  retry_delay: 1                      # Delay between retries (seconds)
```

### Custom System Prompt

```yaml
agent:
  system_prompt: |
    You are an AI assistant specialized in software development.
    You have access to various tools for file operations, calculations,
    and system information. Always be helpful and accurate.
    
    Guidelines:
    - Be concise but thorough
    - Use tools when appropriate
    - Ask for clarification if needed
    - Provide examples when helpful
```

## Audio Configuration

### Basic Audio Settings

```yaml
audio:
  sample_rate: 16000            # Recommended: 16000 or 22050
  channels: 1                   # 1=mono (recommended), 2=stereo
  chunk_size: 1600              # Samples per chunk (100ms at 16kHz)
  buffer_size: 8192             # Audio buffer size
```

### Device Selection

```yaml
audio:
  input_device: null            # null=auto-detect, or device index
  output_device: null           # null=auto-detect, or device index
```

Find device indices:
```bash
poetry run transcriber start --list-devices
```

### Audio Processing

```yaml
audio:
  vad_threshold: 0.5            # Voice detection sensitivity:
                                # - 0.1: Very sensitive
                                # - 0.5: Balanced (recommended)
                                # - 0.9: Less sensitive
  
  echo_cancellation: true       # Prevent feedback loops
  noise_suppression: true       # Reduce background noise
  auto_gain_control: true       # Normalize audio levels
```

## Speech Processing

### Whisper (STT) Configuration

```yaml
whisper:
  model: "tiny"                 # Model size vs accuracy:
                                # - tiny: Fastest, least accurate
                                # - base: Good balance
                                # - small: Better accuracy
                                # - medium: High accuracy
                                # - large: Best accuracy, slowest
  
  language: "en"                # Language codes:
                                # - en: English
                                # - es: Spanish
                                # - fr: French
                                # - de: German
                                # - auto: Auto-detect
  
  compute_type: "float16"       # Precision vs speed:
                                # - int8: Fastest, lower quality
                                # - float16: Balanced (recommended)
                                # - float32: Highest quality, slowest
  
  device: "cpu"                 # Processing device:
                                # - cpu: CPU processing
                                # - cuda: GPU acceleration (if available)
                                # - auto: Automatic selection
```

### Advanced Whisper Settings

```yaml
whisper:
  beam_size: 5                  # Beam search width (1-10)
  best_of: 5                    # Number of candidates (1-10)
  temperature: 0.0              # Sampling randomness (0.0-1.0)
  condition_on_previous_text: true  # Use context from previous audio
  no_speech_threshold: 0.6      # Silence detection threshold
  logprob_threshold: -1.0       # Confidence threshold
  compression_ratio_threshold: 2.4  # Repetition detection
```

### TTS Configuration

```yaml
tts:
  engine: "edge-tts"            # TTS engines:
                                # - edge-tts: High quality, many voices
                                # - gtts: Google TTS (requires internet)
                                # - piper: Local TTS (requires setup)
  
  voice: "en-US-AriaNeural"     # Voice selection (engine-specific)
  rate: "+0%"                   # Speech rate (-50% to +100%)
  volume: "+0%"                 # Volume (-50% to +50%)
  pitch: "+0Hz"                 # Pitch (-200Hz to +200Hz)
  quality: "high"               # Quality (low, medium, high)
```

### Popular Voice Options

**Edge TTS Voices**:
```yaml
# English voices
voice: "en-US-AriaNeural"      # Female, clear
voice: "en-US-GuyNeural"       # Male, professional
voice: "en-US-JennyNeural"     # Female, friendly
voice: "en-GB-SoniaNeural"     # British female
voice: "en-AU-NatashaNeural"   # Australian female

# Other languages
voice: "es-ES-ElviraNeural"    # Spanish female
voice: "fr-FR-DeniseNeural"    # French female
voice: "de-DE-KatjaNeural"     # German female
```

## Performance Settings

### Monitoring Configuration

```yaml
performance:
  enable_monitoring: true       # Enable performance tracking
  metrics_interval: 5           # Collection interval (seconds)
  enable_profiling: false       # Detailed profiling (impacts performance)
  max_history: 1000            # Maximum stored metrics
```

### Performance Thresholds

```yaml
performance:
  cpu_threshold: 80            # CPU warning threshold (%)
  memory_threshold: 85         # Memory warning threshold (%)
  latency_threshold: 1000      # Latency warning threshold (ms)
  auto_optimization: true      # Enable automatic optimizations
```

### Optimization Settings

```yaml
performance:
  model_caching: true          # Cache loaded models
  response_caching: false      # Cache AI responses (not recommended)
  audio_buffering: true        # Enable audio buffering
  concurrent_processing: true  # Enable parallel processing
```

## Session Management

### Storage Configuration

```yaml
session:
  auto_save: true              # Automatically save conversations
  retention_days: 30           # Keep sessions for N days (0=forever)
  max_sessions: 1000          # Maximum stored sessions
  compression: true            # Compress stored data
```

### Backup Settings

```yaml
session:
  backup_enabled: true         # Enable automatic backups
  backup_interval: 24          # Backup interval (hours)
  backup_location: "~/.transcriber/backups"  # Backup directory
  max_backups: 7              # Maximum backup files to keep
```

### Export Configuration

```yaml
session:
  export_format: "json"        # Default export format:
                               # - json: Structured data
                               # - txt: Plain text
                               # - markdown: Formatted text
                               # - csv: Spreadsheet format
  
  include_metadata: true       # Include session metadata in exports
  include_timestamps: true     # Include message timestamps
```

## Tool Configuration

### Tool Categories

```yaml
tools:
  enabled_categories:          # Enable/disable tool categories
    - system                   # File operations, process management
    - utility                  # Calculator, text processing
    - information             # Web search, documentation
    - productivity            # Notes, tasks, timers
```

### Security Settings

```yaml
tools:
  permission_prompts: true     # Prompt before dangerous operations
  sandbox_enabled: true       # Enable tool sandboxing
  timeout: 30                 # Tool execution timeout (seconds)
  max_concurrent: 3           # Maximum concurrent executions
```

### File System Access

```yaml
tools:
  allowed_paths:              # Allowed file system paths
    - "."                     # Current directory
    - "~/Documents"           # Documents folder
    - "~/Downloads"           # Downloads folder
    - "/tmp"                  # Temporary files
  
  blocked_commands:           # Blocked system commands
    - "rm -rf"                # Dangerous deletions
    - "sudo"                  # Privilege escalation
    - "chmod 777"             # Insecure permissions
    - "dd"                    # Disk operations
```

## Environment Variables

### Complete Environment Variable Reference

```bash
# Agent Configuration
TRANSCRIBER_AGENT__MODEL=llama3.2:3b
TRANSCRIBER_AGENT__TEMPERATURE=0.7
TRANSCRIBER_AGENT__MAX_TOKENS=2048
TRANSCRIBER_AGENT__BASE_URL=http://localhost:11434
TRANSCRIBER_AGENT__TIMEOUT=30

# Audio Configuration
TRANSCRIBER_AUDIO__SAMPLE_RATE=16000
TRANSCRIBER_AUDIO__CHUNK_SIZE=1600
TRANSCRIBER_AUDIO__INPUT_DEVICE=0
TRANSCRIBER_AUDIO__OUTPUT_DEVICE=0
TRANSCRIBER_AUDIO__VAD_THRESHOLD=0.5

# Speech Processing
TRANSCRIBER_WHISPER__MODEL=tiny
TRANSCRIBER_WHISPER__LANGUAGE=en
TRANSCRIBER_WHISPER__DEVICE=cpu
TRANSCRIBER_TTS__ENGINE=edge-tts
TRANSCRIBER_TTS__VOICE=en-US-AriaNeural

# Performance
TRANSCRIBER_PERFORMANCE__ENABLE_MONITORING=true
TRANSCRIBER_PERFORMANCE__METRICS_INTERVAL=5
TRANSCRIBER_PERFORMANCE__AUTO_OPTIMIZATION=true

# Session Management
TRANSCRIBER_SESSION__AUTO_SAVE=true
TRANSCRIBER_SESSION__RETENTION_DAYS=30
TRANSCRIBER_SESSION__BACKUP_ENABLED=true

# Tools
TRANSCRIBER_TOOLS__PERMISSION_PROMPTS=true
TRANSCRIBER_TOOLS__TIMEOUT=30
TRANSCRIBER_TOOLS__SANDBOX_ENABLED=true

# Logging
TRANSCRIBER_LOGGING__LEVEL=INFO
TRANSCRIBER_LOGGING__CONSOLE_OUTPUT=true

# UI
TRANSCRIBER_UI__THEME=dark
TRANSCRIBER_UI__SHOW_TIMESTAMPS=true
```

### Using .env File

Create `.env` file in project root:

```bash
# .env file for AI Voice Agent
TRANSCRIBER_AGENT__MODEL=llama3.2:3b
TRANSCRIBER_AGENT__TEMPERATURE=0.7
TRANSCRIBER_AUDIO__SAMPLE_RATE=22050
TRANSCRIBER_WHISPER__MODEL=base
TRANSCRIBER_TTS__ENGINE=edge-tts
TRANSCRIBER_PERFORMANCE__ENABLE_MONITORING=true
```

## Advanced Configuration

### Custom Model Configuration

```yaml
agent:
  custom_models:
    fast:
      model: "llama3.2:1b"
      temperature: 0.5
      max_tokens: 1024
    
    quality:
      model: "llama3.2:7b"
      temperature: 0.7
      max_tokens: 4096
    
    creative:
      model: "qwen2.5:3b"
      temperature: 0.9
      max_tokens: 2048
```

Use custom configurations:
```bash
poetry run transcriber start --profile fast
poetry run transcriber start --profile quality
```

### Multi-Language Support

```yaml
languages:
  default: "en"
  supported:
    - code: "en"
      name: "English"
      whisper_model: "base"
      tts_voice: "en-US-AriaNeural"
    
    - code: "es"
      name: "Spanish"
      whisper_model: "base"
      tts_voice: "es-ES-ElviraNeural"
    
    - code: "fr"
      name: "French"
      whisper_model: "base"
      tts_voice: "fr-FR-DeniseNeural"
```

### Development Configuration

```yaml
development:
  debug_mode: true
  mock_audio: false            # Use mock audio for testing
  mock_llm: false             # Use mock LLM responses
  test_mode: false            # Enable test mode
  profiling_enabled: true     # Enable detailed profiling
  
  # Override settings for development
  agent:
    model: "llama3.2:1b"      # Faster model for development
    temperature: 0.5
  
  audio:
    sample_rate: 16000        # Lower quality for faster processing
  
  whisper:
    model: "tiny"             # Fastest model for development
```

### Production Configuration

```yaml
production:
  # Optimized settings for production use
  agent:
    model: "llama3.2:3b"      # Balanced model
    temperature: 0.7
    max_tokens: 2048
  
  audio:
    sample_rate: 22050        # Higher quality audio
    echo_cancellation: true
    noise_suppression: true
  
  whisper:
    model: "base"             # Good accuracy/speed balance
    compute_type: "float16"
  
  performance:
    enable_monitoring: true
    auto_optimization: true
    cpu_threshold: 70
    memory_threshold: 80
  
  session:
    auto_save: true
    backup_enabled: true
    retention_days: 90
  
  logging:
    level: "INFO"
    file: "/var/log/transcriber/transcriber.log"
```

### Configuration Validation

Validate your configuration:

```bash
# Check configuration syntax
poetry run transcriber configure --validate

# Test configuration
poetry run transcriber configure --test

# Show current configuration
poetry run transcriber configure --show
```

### Configuration Templates

Generate configuration templates:

```bash
# Generate minimal configuration
poetry run transcriber configure --template minimal

# Generate full configuration with all options
poetry run transcriber configure --template full

# Generate development configuration
poetry run transcriber configure --template development

# Generate production configuration
poetry run transcriber configure --template production
```

---

This configuration guide covers all available settings. For specific use cases, see:

- [Performance Guide](PERFORMANCE.md) for optimization settings
- [Troubleshooting Guide](TROUBLESHOOTING.md) for configuration issues
- [User Guide](USER_GUIDE.md) for usage examples