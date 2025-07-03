#!/usr/bin/env python3
"""
Simple voice pipeline demo launcher.
"""

import asyncio
from transcriber.simple_voice import run_simple_voice_demo
from transcriber.config import settings

if __name__ == "__main__":
    asyncio.run(run_simple_voice_demo(settings))