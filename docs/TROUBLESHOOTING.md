# Troubleshooting Guide

Common issues and solutions for the AI Voice Agent.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Installation Issues](#installation-issues)
3. [Audio Problems](#audio-problems)
4. [Ollama Issues](#ollama-issues)
5. [Performance Issues](#performance-issues)
6. [Tool Execution Problems](#tool-execution-problems)
7. [Session Management Issues](#session-management-issues)
8. [Configuration Problems](#configuration-problems)
9. [Error Messages](#error-messages)
10. [Getting Help](#getting-help)

## Quick Diagnostics

### System Health Check

Run these commands to quickly diagnose common issues:

```bash
# Check if transcriber is installed
poetry run transcriber --help

# Test Ollama connection
ollama list

# Test text mode (no audio needed)
poetry run transcriber chat

# Check audio devices
poetry run transcriber start --list-devices

# Check performance
poetry run transcriber performance
```

### Common Quick Fixes

1. **Restart Ollama**: `ollama serve`
2. **Update models**: `ollama pull llama3.2:3b`
3. **Clear sessions**: `poetry run transcriber cleanup --dry-run`
4. **Reset config**: Delete `~/.transcriber/config.yaml`
5. **Reinstall dependencies**: `poetry install --sync`

## Installation Issues

### Poetry Installation Problems

**Problem**: `poetry: command not found`

**Solution**:
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.local/bin:$PATH"

# Reload shell
source ~/.bashrc  # or ~/.zshrc
```

**Problem**: `poetry install` fails with dependency conflicts

**Solution**:
```bash
# Clear poetry cache
poetry cache clear --all pypi

# Remove lock file and reinstall
rm poetry.lock
poetry install

# If still failing, try without dev dependencies
poetry install --only main
```

### Python Version Issues

**Problem**: `Python 3.10+ required`

**Solution**:
```bash
# Check Python version
python --version

# Install Python 3.10+ using pyenv
curl https://pyenv.run | bash
pyenv install 3.11.0
pyenv global 3.11.0

# Or use system package manager
# macOS: brew install python@3.11
# Ubuntu: sudo apt install python3.11
```

### Dependency Installation Failures

**Problem**: `faster-whisper` or audio dependencies fail to install

**Solution**:
```bash
# Install system dependencies first
# macOS:
brew install ffmpeg pkg-config

# Ubuntu/Debian:
sudo apt update
sudo apt install ffmpeg libavcodec-dev libavformat-dev libavutil-dev pkg-config

# Then reinstall
poetry install
```

**Problem**: `webrtcvad` compilation fails

**Solution**:
```bash
# Install build tools
# macOS:
xcode-select --install

# Ubuntu/Debian:
sudo apt install build-essential

# Alternative: Use conda
conda install -c conda-forge webrtcvad
```

## Audio Problems

### No Audio Devices Found

**Problem**: `No audio devices available`

**Solution**:
```bash
# List system audio devices
# macOS:
system_profiler SPAudioDataType

# Linux:
arecord -l  # Input devices
aplay -l    # Output devices

# Check permissions (Linux)
sudo usermod -a -G audio $USER
# Logout and login again

# Test with transcriber
poetry run transcriber start --list-devices
```

### Microphone Not Working

**Problem**: Audio input not detected

**Solution**:
1. **Check microphone permissions**:
   - macOS: System Preferences → Security & Privacy → Microphone
   - Linux: Check PulseAudio/ALSA settings

2. **Test microphone**:
   ```bash
   # Record test audio (macOS/Linux)
   # Press Ctrl+C to stop
   poetry run python -c "
   import sounddevice as sd
   import numpy as np
   print('Recording... Press Ctrl+C to stop')
   recording = sd.rec(int(5 * 16000), samplerate=16000, channels=1)
   sd.wait()
   print(f'Recorded {len(recording)} samples')
   "
   ```

3. **Use specific device**:
   ```bash
   # List devices and use specific index
   poetry run transcriber start --list-devices
   poetry run transcriber start --device 2
   ```

### Audio Feedback/Echo

**Problem**: AI hears its own voice

**Solution**:
1. **Use headphones** instead of speakers
2. **Adjust audio settings**:
   ```bash
   # Set environment variables
   export TRANSCRIBER_AUDIO__ECHO_CANCELLATION=true
   export TRANSCRIBER_AUDIO__VAD_THRESHOLD=0.7
   ```
3. **Use different devices**:
   ```bash
   # Use USB headset for both input and output
   poetry run transcriber start --device 1
   ```

### Poor Audio Quality

**Problem**: Speech recognition accuracy is low

**Solution**:
1. **Improve audio settings**:
   ```bash
   export TRANSCRIBER_AUDIO__SAMPLE_RATE=22050
   export TRANSCRIBER_WHISPER__MODEL=base  # Better than tiny
   ```

2. **Reduce background noise**:
   - Use in quiet environment
   - Use noise-canceling microphone
   - Adjust VAD threshold

3. **Test different models**:
   ```bash
   # Try different Whisper models
   poetry run transcriber start --model llama3.2:1b  # Faster
   ```

## Ollama Issues

### Ollama Not Running

**Problem**: `Connection refused` or `Ollama server not available`

**Solution**:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Start Ollama service
ollama serve

# Or start as background service
# macOS/Linux:
nohup ollama serve > /dev/null 2>&1 &

# Check process
ps aux | grep ollama
```

### Model Not Found

**Problem**: `Model 'llama3.2:3b' not found`

**Solution**:
```bash
# List available models
ollama list

# Pull required model
ollama pull llama3.2:3b

# Try alternative models
ollama pull llama3.2:1b  # Smaller, faster
ollama pull qwen2.5:3b   # Alternative model

# Use different model
poetry run transcriber start --model llama3.2:1b
```

### Ollama Performance Issues

**Problem**: Slow response times or high memory usage

**Solution**:
1. **Use smaller model**:
   ```bash
   ollama pull llama3.2:1b
   poetry run transcriber start --model llama3.2:1b
   ```

2. **Adjust Ollama settings**:
   ```bash
   # Set memory limit
   export OLLAMA_MAX_LOADED_MODELS=1
   export OLLAMA_NUM_PARALLEL=1
   
   # Restart Ollama
   pkill ollama
   ollama serve
   ```

3. **Monitor resources**:
   ```bash
   # Check memory usage
   poetry run transcriber performance --component agent
   ```

## Performance Issues

### High Latency

**Problem**: Response times > 2 seconds

**Solution**:
1. **Check component performance**:
   ```bash
   poetry run transcriber performance --detailed
   poetry run transcriber profile --duration 30
   ```

2. **Optimize settings**:
   ```bash
   # Use faster models
   export TRANSCRIBER_WHISPER__MODEL=tiny
   export TRANSCRIBER_AGENT__MODEL=llama3.2:1b
   
   # Reduce audio chunk size
   export TRANSCRIBER_AUDIO__CHUNK_SIZE=800
   ```

3. **System optimization**:
   ```bash
   # Close unnecessary applications
   # Ensure sufficient RAM (8GB+ recommended)
   # Use SSD storage for better I/O
   ```

### High Memory Usage

**Problem**: System running out of memory

**Solution**:
1. **Monitor memory usage**:
   ```bash
   poetry run transcriber performance --component memory
   ```

2. **Use smaller models**:
   ```bash
   # Smaller Whisper model
   export TRANSCRIBER_WHISPER__MODEL=tiny
   
   # Smaller LLM model
   export TRANSCRIBER_AGENT__MODEL=llama3.2:1b
   ```

3. **Limit session history**:
   ```bash
   # Clean old sessions
   poetry run transcriber cleanup --days 7
   
   # Limit conversation memory
   export TRANSCRIBER_AGENT__MAX_TOKENS=1024
   ```

### CPU Usage Issues

**Problem**: High CPU usage affecting system performance

**Solution**:
1. **Adjust processing settings**:
   ```bash
   # Reduce audio processing frequency
   export TRANSCRIBER_AUDIO__SAMPLE_RATE=16000
   export TRANSCRIBER_PERFORMANCE__METRICS_INTERVAL=10
   ```

2. **Use hardware acceleration** (if available):
   ```bash
   # GPU acceleration for Whisper (requires CUDA)
   export TRANSCRIBER_WHISPER__DEVICE=cuda
   ```

## Tool Execution Problems

### Tools Not Working

**Problem**: "Tool not found" or tool execution fails

**Solution**:
1. **Check tool availability**:
   ```bash
   poetry run transcriber list-tools
   poetry run transcriber list-tools --detailed
   ```

2. **Verify tool permissions**:
   ```bash
   # Check file permissions for file operations
   ls -la /path/to/file
   
   # Check system permissions
   # Some tools may require specific permissions
   ```

3. **Test tools individually**:
   ```bash
   # Test in text mode
   poetry run transcriber chat
   # Then try: "Use the calculator to compute 2+2"
   ```

### Permission Errors

**Problem**: "Permission denied" when using tools

**Solution**:
1. **File permission issues**:
   ```bash
   # Fix file permissions
   chmod 644 /path/to/file  # Read/write for owner
   chmod 755 /path/to/directory  # Execute for directory
   ```

2. **System permission issues**:
   ```bash
   # Add user to required groups (Linux)
   sudo usermod -a -G sudo $USER  # For system tools
   ```

3. **Configure tool permissions**:
   ```yaml
   # In ~/.transcriber/config.yaml
   tools:
     permission_prompts: true
     enabled_categories:
       - system
       - utility
   ```

## Session Management Issues

### Database Errors

**Problem**: "Database locked" or session storage errors

**Solution**:
1. **Check database file**:
   ```bash
   # Check if database file exists and is writable
   ls -la data/sessions.db
   
   # Fix permissions
   chmod 644 data/sessions.db
   ```

2. **Reset database**:
   ```bash
   # Backup existing sessions
   cp data/sessions.db data/sessions.db.backup
   
   # Remove corrupted database (will recreate)
   rm data/sessions.db
   
   # Restart transcriber
   poetry run transcriber start
   ```

3. **Clean up sessions**:
   ```bash
   # Clean old sessions
   poetry run transcriber cleanup --days 30
   ```

### Session Export Issues

**Problem**: Export fails or produces empty files

**Solution**:
1. **Check session exists**:
   ```bash
   poetry run transcriber list-sessions
   poetry run transcriber replay abc12345
   ```

2. **Try different formats**:
   ```bash
   # Try different export formats
   poetry run transcriber export --format txt
   poetry run transcriber export --format json
   ```

3. **Check file permissions**:
   ```bash
   # Ensure write permissions in target directory
   ls -la /path/to/export/directory
   ```

## Configuration Problems

### Config File Issues

**Problem**: Configuration not loading or invalid settings

**Solution**:
1. **Check config file location**:
   ```bash
   # Default locations
   ls -la ~/.transcriber/config.yaml
   ls -la ./config.yaml
   ```

2. **Validate YAML syntax**:
   ```bash
   # Use Python to validate YAML
   python -c "import yaml; yaml.safe_load(open('~/.transcriber/config.yaml'))"
   ```

3. **Reset to defaults**:
   ```bash
   # Backup current config
   mv ~/.transcriber/config.yaml ~/.transcriber/config.yaml.backup
   
   # Generate new config
   poetry run transcriber configure
   ```

### Environment Variables

**Problem**: Environment variables not taking effect

**Solution**:
1. **Check variable names**:
   ```bash
   # Correct format: TRANSCRIBER_SECTION__SETTING
   export TRANSCRIBER_AGENT__MODEL=llama3.2:1b
   export TRANSCRIBER_AUDIO__SAMPLE_RATE=22050
   
   # Verify variables are set
   env | grep TRANSCRIBER
   ```

2. **Use .env file**:
   ```bash
   # Create .env file in project root
   cat > .env << EOF
   TRANSCRIBER_AGENT__MODEL=llama3.2:1b
   TRANSCRIBER_AUDIO__SAMPLE_RATE=22050
   EOF
   ```

## Error Messages

### Common Error Messages and Solutions

#### `ModuleNotFoundError: No module named 'transcriber'`

**Solution**:
```bash
# Ensure you're in the project directory
cd /path/to/transcriber

# Use poetry run
poetry run transcriber start

# Or activate virtual environment
poetry shell
transcriber start
```

#### `RuntimeError: No audio backend available`

**Solution**:
```bash
# Install audio dependencies
# macOS:
brew install portaudio

# Ubuntu/Debian:
sudo apt install portaudio19-dev

# Reinstall sounddevice
poetry add sounddevice --force
```

#### `ConnectionError: Ollama server not available`

**Solution**:
```bash
# Start Ollama
ollama serve

# Check if running
curl http://localhost:11434/api/version

# Use different URL if needed
export TRANSCRIBER_AGENT__BASE_URL=http://localhost:11434
```

#### `FileNotFoundError: Model file not found`

**Solution**:
```bash
# Download required models
ollama pull llama3.2:3b

# For Whisper models, they download automatically
# But you can pre-download:
python -c "
import whisper
whisper.load_model('tiny')
"
```

#### `PermissionError: Access denied`

**Solution**:
```bash
# Fix file permissions
chmod -R 755 ~/.transcriber/

# Fix data directory permissions
chmod -R 644 data/

# On Linux, check SELinux/AppArmor if applicable
```

## Getting Help

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Set debug environment variable
export TRANSCRIBER_DEBUG=true
export TRANSCRIBER_LOG_LEVEL=DEBUG

# Run with debug output
poetry run transcriber start 2>&1 | tee debug.log
```

### Log Files

Check log files for detailed error information:

```bash
# Default log locations
tail -f ~/.transcriber/logs/transcriber.log
tail -f ./transcriber.log

# System logs (Linux)
journalctl -u transcriber --follow
```

### Performance Diagnostics

Run comprehensive diagnostics:

```bash
# System performance
poetry run transcriber performance --detailed

# Component profiling
poetry run transcriber profile --duration 60

# Benchmark tests
poetry run transcriber benchmark --output benchmark.json
```

### Collecting Debug Information

When reporting issues, include:

1. **System Information**:
   ```bash
   # System details
   uname -a
   python --version
   poetry --version
   
   # Transcriber version
   poetry run transcriber --version
   
   # Ollama version
   ollama --version
   ```

2. **Configuration**:
   ```bash
   # Environment variables
   env | grep TRANSCRIBER
   
   # Config file (remove sensitive data)
   cat ~/.transcriber/config.yaml
   ```

3. **Error logs**:
   ```bash
   # Recent logs
   tail -100 ~/.transcriber/logs/transcriber.log
   ```

4. **Performance metrics**:
   ```bash
   # Current performance
   poetry run transcriber performance --detailed
   ```

### Community Support

- **GitHub Issues**: [Report bugs and request features](https://github.com/yourusername/transcriber/issues)
- **GitHub Discussions**: [Ask questions and share tips](https://github.com/yourusername/transcriber/discussions)
- **Documentation**: [Complete documentation](docs/)

### Professional Support

For enterprise users or complex deployments:

- Custom configuration assistance
- Performance optimization consulting
- Custom tool development
- Integration support

Contact: [support@transcriber.ai](mailto:support@transcriber.ai)

---

If you can't find a solution to your problem in this guide, please [open an issue](https://github.com/yourusername/transcriber/issues) with:

1. Detailed description of the problem
2. Steps to reproduce
3. Error messages and logs
4. System information
5. Configuration details

We'll help you resolve the issue as quickly as possible!