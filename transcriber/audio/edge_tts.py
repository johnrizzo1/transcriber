"""
Microsoft Edge Text-to-Speech module for voice synthesis.
"""

import asyncio
import io
import logging
from collections.abc import AsyncGenerator
from typing import Optional

import numpy as np
import edge_tts

from ..config import Settings

logger = logging.getLogger(__name__)


class EdgeTTSEngine:
    """Text-to-Speech engine using Microsoft Edge TTS."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        # Popular Edge TTS voices:
        # Female: en-US-AriaNeural, en-US-JennyNeural, en-GB-SoniaNeural
        # Male: en-US-GuyNeural, en-US-EricNeural, en-GB-RyanNeural
        self.voice = getattr(settings, 'edge_voice', "en-US-AriaNeural")
        self.rate = "+0%"  # Normal speed (-50% to +50%)
        self.volume = "+0%"  # Normal volume  
        self.pitch = "+0Hz"  # Normal pitch
        
    async def initialize(self) -> None:
        """Initialize the TTS engine."""
        logger.info("Initializing Edge TTS engine...")
        
        # List available voices
        try:
            voices = await edge_tts.list_voices()
            en_voices = [v for v in voices if v["Locale"].startswith("en-")]
            logger.info(f"Found {len(en_voices)} English voices")
            
            # You can change the voice here if desired
            # self.voice = "en-US-GuyNeural"  # Male voice
            # self.voice = "en-GB-SoniaNeural"  # British female
            
            logger.info(f"Using voice: {self.voice}")
            
        except Exception as e:
            logger.error(f"Failed to list Edge TTS voices: {e}")
            
    async def synthesize_text(self, text: str) -> np.ndarray:
        """
        Synthesize text to audio using Edge TTS.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio data as numpy array
        """
        if not text.strip():
            return np.array([], dtype=np.float32)
        
        try:
            # Create TTS communicator
            communicate = edge_tts.Communicate(
                text,
                self.voice,
                rate=self.rate,
                volume=self.volume,
                pitch=self.pitch
            )
            
            # Collect audio chunks
            audio_chunks = []
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])
            
            if not audio_chunks:
                logger.warning("No audio chunks received from Edge TTS")
                return np.array([], dtype=np.float32)
            
            # Combine all chunks
            audio_bytes = b"".join(audio_chunks)
            
            # Edge TTS returns MP3 data, convert using pydub if available
            try:
                from pydub import AudioSegment
                
                # Load MP3 data
                audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
                
                # Convert to mono if stereo
                if audio_segment.channels > 1:
                    audio_segment = audio_segment.set_channels(1)
                
                # Convert to 16kHz for compatibility
                audio_segment = audio_segment.set_frame_rate(16000)
                
                # Convert to numpy array
                samples = np.array(audio_segment.get_array_of_samples())
                audio_data = samples.astype(np.float32) / 32768.0
                
                logger.debug(f"Synthesized {len(text)} chars to {len(audio_data)} samples")
                return audio_data
                
            except ImportError:
                logger.error("pydub not available for audio conversion")
                # Return silent audio as fallback
                return np.zeros(16000, dtype=np.float32)  # 1 second of silence
                
        except Exception as e:
            logger.error(f"Edge TTS synthesis error: {e}")
            return np.array([], dtype=np.float32)
    
    async def synthesize_stream(self, text_stream: AsyncGenerator[str, None]) -> AsyncGenerator[np.ndarray, None]:
        """
        Synthesize streaming text to audio.
        
        Args:
            text_stream: Async generator of text chunks
            
        Yields:
            Audio chunks
        """
        sentence_buffer = ""
        
        async for text_chunk in text_stream:
            sentence_buffer += text_chunk
            
            # Check for sentence endings
            sentences = []
            current_sentence = ""
            
            for char in sentence_buffer:
                current_sentence += char
                if char in '.!?':
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
            
            # Process complete sentences
            for sentence in sentences:
                if sentence:
                    audio_data = await self.synthesize_text(sentence)
                    if len(audio_data) > 0:
                        yield audio_data
            
            # Keep remaining text for next iteration
            sentence_buffer = current_sentence
        
        # Process any remaining text
        if sentence_buffer.strip():
            audio_data = await self.synthesize_text(sentence_buffer.strip())
            if len(audio_data) > 0:
                yield audio_data
    
    def get_voice_info(self) -> dict:
        """Get information about the current voice."""
        return {
            "engine": "Edge TTS",
            "voice": self.voice,
            "sample_rate": 16000,
            "format": "mp3 (converted)"
        }
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Edge TTS engine cleaned up")


class EdgeTTSService:
    """High-level Edge TTS service for easy integration."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine = EdgeTTSEngine(settings)
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
    
    async def speak_stream(self, text_stream: AsyncGenerator[str, None]) -> AsyncGenerator[np.ndarray, None]:
        """Synthesize streaming text to speech."""
        if not self._initialized:
            await self.initialize()
        async for audio_chunk in self.engine.synthesize_stream(text_stream):
            yield audio_chunk
    
    def get_voice_info(self) -> dict:
        """Get voice information."""
        return self.engine.get_voice_info()
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.engine.cleanup()
        self._initialized = False