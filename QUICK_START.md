# Transcriber Quick Start

## 🚀 **Get Started in 30 Seconds**

### **Show Help**
```bash
# Main command (recommended)
poetry run transcriber

# Alternative
poetry run python run_transcriber.py
```

### **Try the AI Agent**
```bash
# Text demo with sample questions
poetry run transcriber demo

# Voice pipeline demo (Mock STT + Google TTS)
poetry run transcriber voice

# Interactive chat
poetry run transcriber chat
```

### **Check Audio Setup**
```bash
# List microphones and speakers
poetry run transcriber devices
```

## ✅ **What Works Right Now**

- **🤖 AI Agent**: Full conversation capabilities with Ollama
- **💬 Text Chat**: Interactive conversations
- **🎤 Voice Pipeline**: Complete STT→Agent→TTS flow (using mock STT + Google TTS)
- **🎵 Audio System**: Device detection and audio processing
- **📊 Rich Interface**: Beautiful terminal output

## 🎯 **Most Useful Commands**

| Command | What It Does |
|---------|--------------|
| `poetry run transcriber demo` | **Best first test** - Shows AI in action |
| `poetry run transcriber voice` | **Voice demo** - Complete pipeline |
| `poetry run transcriber chat` | **Interactive** - Talk with the AI |
| `poetry run transcriber devices` | **Audio setup** - Check microphones |

## 🔧 **Configuration**

Default settings work out of the box. To customize:

```bash
# Use smaller/faster model
export TRANSCRIBER_AGENT__MODEL="llama3.2:1b"

# More creative responses
export TRANSCRIBER_AGENT__TEMPERATURE=0.9
```

## 🎉 **You're Ready!**

Your transcriber voice agent is **working and ready to use**. Start with:

```bash
poetry run transcriber demo
```

Then explore the other commands. See `USAGE_GUIDE.md` for complete documentation.