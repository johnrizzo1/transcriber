"""
Simple Text-to-Speech using system commands.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import Settings

logger = logging.getLogger(__name__)


class SimpleTTSEngine:
    """Simple TTS engine using system 'say' command (macOS) or espeak (Linux)."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.platform = None
        self.temp_dir = Path(tempfile.gettempdir()) / "transcriber_tts"
        self.temp_dir.mkdir(exist_ok=True)
        
    async def initialize(self) -> None:
        """Initialize the TTS engine."""
        import platform
        self.platform = platform.system().lower()
        
        if self.platform == "darwin":
            logger.info("Using macOS 'say' command for TTS")
        elif self.platform == "linux":
            logger.info("Using espeak for TTS")
        else:
            logger.warning(f"Platform {self.platform} may not support TTS")
            
    async def synthesize_text(self, text: str) -> np.ndarray:
        """
        Synthesize text to audio using system TTS.
        
        For macOS, this uses the 'say' command.
        For Linux, this uses espeak.
        
        Returns dummy audio data since we play directly.
        """
        if not text.strip():
            return np.array([], dtype=np.float32)
        
        # Clean text for TTS
        text = text.replace('"', "'")  # Replace quotes to avoid command issues
        
        try:
            if self.platform == "darwin":
                # Use macOS say command
                cmd = ["say", "-v", "Samantha", text]
            elif self.platform == "linux":
                # Use espeak
                cmd = ["espeak", text]
            else:
                logger.warning("No TTS available for this platform")
                return np.zeros(16000, dtype=np.float32)  # 1 second silence
            
            # Run TTS command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
            # Return dummy audio since we played it directly
            # The duration is approximate based on text length
            duration = len(text) * 0.05  # Rough estimate
            samples = int(16000 * duration)
            return np.zeros(samples, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return np.array([], dtype=np.float32)
    
    def get_voice_info(self) -> dict:
        """Get information about the current voice."""
        return {
            "engine": f"System TTS ({self.platform})",
            "voice": "Samantha" if self.platform == "darwin" else "default",
            "sample_rate": 16000,
            "format": "direct playback"
        }
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        pass


class SimpleTTSService:
    """Simple TTS service using system commands."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine = SimpleTTSEngine(settings)
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize the TTS service."""
        if not self._initialized:
            await self.engine.initialize()
            self._initialized = True
            
    async def speak(self, text: str) -> np.ndarray:
        """Synthesize text to speech."""
        if not self._initialized:
            await self.initialize()
        return await self.engine.synthesize_text(text)
    
    async def speak_stream(self, text_stream) -> None:
        """Not implemented for simple TTS."""
        # Collect all text first
        full_text = ""
        async for chunk in text_stream:
            full_text += chunk
        
        if full_text:
            await self.speak(full_text)
    
    def get_voice_info(self) -> dict:
        """Get voice information."""
        return self.engine.get_voice_info()
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.engine.cleanup()
        self._initialized = False