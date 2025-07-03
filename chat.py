#!/usr/bin/env python3
"""
Simple chat interface for testing the transcriber agent.
"""

import asyncio
from transcriber.config import settings
from transcriber.agent.text_agent import run_interactive_chat

if __name__ == "__main__":
    asyncio.run(run_interactive_chat(settings))