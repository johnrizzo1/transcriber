# Transcriber Usage Guide

## 🚀 **Quick Start - Choose Your Interface**

You now have **two ways** to run the transcriber:

### **Option 1: Simple Launcher (Recommended)**
```bash
# Show help
poetry run python run_transcriber.py

# Run demos
poetry run python run_transcriber.py demo    # Text-only agent demo
poetry run python run_transcriber.py voice   # Voice pipeline demo
poetry run python run_transcriber.py chat    # Interactive chat
poetry run python run_transcriber.py devices # List audio devices
```

### **Option 2: Full CLI (Advanced)**
```bash
# Show help
poetry run transcriber

# Available commands
poetry run transcriber start              # Voice agent
poetry run transcriber chat               # Text chat
poetry run transcriber list-tools         # Show tools
poetry run transcriber start --list-devices  # Audio devices
```

## 📋 **Available Commands**

### **Demo Commands**
| Command | Description | Interface |
|---------|-------------|-----------|
| `run_transcriber.py demo` | Text agent demo with sample questions | Simple |
| `run_transcriber.py voice` | Voice pipeline demo (Mock STT + GTTS) | Simple |
| `transcriber start` | Full voice agent (fallback to text) | Advanced |

### **Interactive Commands**
| Command | Description | Interface |
|---------|-------------|-----------|
| `run_transcriber.py chat` | Interactive text chat | Simple |
| `transcriber chat` | Interactive text chat | Advanced |

### **Utility Commands**
| Command | Description | Interface |
|---------|-------------|-----------|
| `run_transcriber.py devices` | List audio devices | Simple |
| `transcriber start --list-devices` | List audio devices | Advanced |
| `transcriber list-tools` | List available tools | Advanced |

## 🎯 **What Each Command Does**

### **🤖 Text Demo** (`demo`)
Shows the AI agent in action with predefined questions:
```
🤖 Transcriber AI Voice Agent Demo
Q1: Hello! What are you?
🤖 Agent: Hello! I'm an artificial intelligence...
```

### **🎤 Voice Demo** (`voice`)
Demonstrates the complete voice pipeline:
```
✅ TTS (GTTS) initialized
✅ Mock STT initialized
🎤 Simulated Voice Input 1: Hello, how are you?
🤖 Agent Response: I'm doing well...
🔊 Generating speech... Generated 562275 audio samples
```

### **💬 Interactive Chat** (`chat`)
Live conversation with the AI agent:
```
🎤 You: What's the weather like?
🤖 Agent: I'm a large language model, I don't have real-time access...
```

### **🎵 Audio Devices** (`devices`)
Shows available microphones and speakers:
```
🎤 Input devices (microphones):
  0: MacBook Pro Microphone (1 channels)
🔊 Output devices (speakers):
  1: MacBook Pro Speakers (2 channels)
```

## ⚡ **Quick Testing**

### **Test Everything Works**
```bash
# 1. Test AI agent
poetry run python run_transcriber.py demo

# 2. Test voice pipeline
poetry run python run_transcriber.py voice

# 3. Check audio devices
poetry run python run_transcriber.py devices
```

### **Development Workflow**
```bash
# Start development session
poetry run python run_transcriber.py chat

# Test voice capabilities
poetry run python run_transcriber.py voice

# Check configuration
poetry run python -c "from transcriber.config import settings; print(settings.model_dump())"
```

## 🔧 **Configuration**

### **Environment Variables**
```bash
# Customize AI model
export TRANSCRIBER_AGENT__MODEL="llama3.2:1b"

# Adjust audio settings
export TRANSCRIBER_AUDIO__SAMPLE_RATE=22050

# Set TTS language
export TRANSCRIBER_PIPER__LANGUAGE="en"
```

### **Direct Configuration**
```python
# Edit transcriber/config.py
settings.agent.model = "llama3.2:1b"    # Smaller model
settings.agent.temperature = 0.9        # More creative
settings.audio.sample_rate = 22050      # Higher quality
```

## 📊 **Performance Tips**

### **Faster Response**
```bash
# Use smaller model for faster responses
export TRANSCRIBER_AGENT__MODEL="llama3.2:1b"

# Lower temperature for more focused responses
export TRANSCRIBER_AGENT__TEMPERATURE=0.3
```

### **Better Quality**
```bash
# Use larger model for better responses
export TRANSCRIBER_AGENT__MODEL="llama3.2:3b"

# Higher audio quality
export TRANSCRIBER_AUDIO__SAMPLE_RATE=44100
```

## 🐛 **Troubleshooting**

### **Command Not Found**
```bash
# Problem: poetry run transcriber fails
# Solution: Use simple launcher
poetry run python run_transcriber.py help
```

### **Ollama Connection**
```bash
# Problem: Failed to connect to Ollama
# Solution: Start Ollama first
ollama serve

# Then test connection
poetry run python -c "import ollama; print(ollama.list())"
```

### **Audio Issues**
```bash
# Problem: No audio devices
# Solution: Check devices first
poetry run python run_transcriber.py devices

# Problem: Audio capture fails
# Solution: Check permissions and devices
```

### **Dependencies Missing**
```bash
# Problem: Module not found
# Solution: Reinstall dependencies
poetry install

# Check installation
poetry show gtts ollama rich
```

## 🎯 **Next Steps**

### **For Development**
1. **Start with text mode**: `poetry run python run_transcriber.py demo`
2. **Test voice pipeline**: `poetry run python run_transcriber.py voice`
3. **Interactive testing**: `poetry run python run_transcriber.py chat`

### **For Production**
1. **Add real STT**: Uncomment `openai-whisper` in `pyproject.toml`
2. **Add offline TTS**: Uncomment `edge-tts` for offline capabilities
3. **Add tools**: Implement tool system for agent capabilities

### **For Customization**
1. **Modify prompts**: Edit system prompts in agent modules
2. **Add commands**: Extend `run_transcriber.py` with new features
3. **Configure models**: Adjust settings for your use case

## 📱 **Example Session**

```bash
# Start a complete testing session
cd /path/to/transcriber

# 1. Check everything works
poetry run python run_transcriber.py demo

# 2. Test voice pipeline
poetry run python run_transcriber.py voice

# 3. Interactive chat
poetry run python run_transcriber.py chat
# Type: "Hello, tell me about yourself"
# Type: "quit" to exit

# 4. Check audio setup
poetry run python run_transcriber.py devices
```

## 🎉 **Success!**

You now have:
- ✅ **Working CLI** with proper help display
- ✅ **Multiple interfaces** (simple + advanced)
- ✅ **Complete voice pipeline** ready for use
- ✅ **Easy testing** with built-in demos
- ✅ **Flexible configuration** for different needs

**Your transcriber voice agent is ready to use!** 🚀