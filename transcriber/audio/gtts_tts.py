"""
Google Text-to-Speech (GTTS) module for voice synthesis.
"""

import asyncio
import io
import logging
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Optional

import numpy as np
from gtts import gTTS

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    AudioSegment = None

from ..config import Settings

logger = logging.getLogger(__name__)


class GTTSEngine:
    """Text-to-Speech engine using Google TTS."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.temp_dir = Path(tempfile.gettempdir()) / "transcriber_gtts"
        self.temp_dir.mkdir(exist_ok=True)
        self.language = getattr(settings.piper, 'language', 'en')  # Default to English
        
    async def initialize(self) -> None:
        """Initialize the TTS engine."""
        logger.info("Initializing GTTS engine...")
        
        # Test GTTS with a simple phrase
        try:
            test_tts = gTTS(text="Hello", lang=self.language)
            logger.info(f"GTTS engine initialized with language: {self.language}")
        except Exception as e:
            logger.error(f"Failed to initialize GTTS: {e}")
            raise
        
    async def synthesize_text(self, text: str) -> np.ndarray:
        """
        Synthesize text to audio using GTTS.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio data as numpy array
        """
        if not text.strip():
            return np.array([], dtype=np.float32)
        
        # Run synthesis in thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        
        def synthesize():
            try:
                # Create TTS object
                tts = gTTS(text=text, lang=self.language, slow=False)
                
                # Save to temporary file
                temp_file = self.temp_dir / f"tts_{id(text)}.mp3"
                tts.save(str(temp_file))
                
                # Load the MP3 file and convert to numpy array
                if PYDUB_AVAILABLE and temp_file.exists():
                    try:
                        # Load MP3 with pydub
                        audio_segment = AudioSegment.from_mp3(str(temp_file))
                        
                        # Convert to mono if stereo
                        if audio_segment.channels > 1:
                            audio_segment = audio_segment.set_channels(1)
                        
                        # Convert to target sample rate (16kHz for compatibility)
                        target_sample_rate = 16000
                        audio_segment = audio_segment.set_frame_rate(target_sample_rate)
                        
                        # Convert to numpy array
                        samples = np.array(audio_segment.get_array_of_samples())
                        
                        # Normalize to float32 [-1, 1]
                        audio_data = samples.astype(np.float32) / 32768.0
                        
                        logger.debug(f"Converted MP3: {len(audio_data)} samples at {target_sample_rate}Hz")
                        
                        # Clean up temp file
                        temp_file.unlink()
                        
                        return audio_data
                        
                    except Exception as e:
                        logger.error(f"Failed to convert MP3: {e}")
                        if temp_file.exists():
                            temp_file.unlink()
                
                # Fallback: Return silent audio
                logger.warning("Using silent audio fallback")
                sample_rate = 16000
                duration = len(text) * 0.1
                num_samples = int(sample_rate * duration)
                audio_data = np.zeros(num_samples, dtype=np.float32)
                
                return audio_data
                
            except Exception as e:
                logger.error(f"GTTS synthesis error: {e}")
                return np.array([], dtype=np.float32)
        
        try:
            audio_data = await loop.run_in_executor(None, synthesize)
            logger.debug(f"Synthesized {len(text)} characters to {len(audio_data)} samples")
            return audio_data
            
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
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
            "engine": "GTTS",
            "language": self.language,
            "sample_rate": 16000,  # We convert to 16kHz
            "format": "mp3 (converted to wav)"
        }
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        # Clean up any remaining temp files
        try:
            for temp_file in self.temp_dir.glob("tts_*.mp3"):
                temp_file.unlink()
        except Exception as e:
            logger.warning(f"Error cleaning up temp files: {e}")
        
        logger.info("GTTS engine cleaned up")


class GTTSProcessor:
    """High-level GTTS processor."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine = GTTSEngine(settings)
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize the TTS processor."""
        if not self._initialized:
            await self.engine.initialize()
            self._initialized = True
            
    async def process_text(self, text: str) -> np.ndarray:
        """
        Process text and return synthesized audio.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio data
        """
        if not self._initialized:
            raise RuntimeError("GTTS processor not initialized")
        
        # Preprocess text (basic cleanup)
        cleaned_text = self._preprocess_text(text)
        
        return await self.engine.synthesize_text(cleaned_text)
    
    async def process_text_stream(self, text_stream: AsyncGenerator[str, None]) -> AsyncGenerator[np.ndarray, None]:
        """
        Process streaming text and yield audio chunks.
        
        Args:
            text_stream: Async generator of text chunks
            
        Yields:
            Audio chunks
        """
        if not self._initialized:
            raise RuntimeError("GTTS processor not initialized")
        
        # Create a preprocessing stream
        async def preprocess_stream():
            async for text_chunk in text_stream:
                cleaned_chunk = self._preprocess_text(text_chunk)
                if cleaned_chunk:
                    yield cleaned_chunk
        
        async for audio_chunk in self.engine.synthesize_stream(preprocess_stream()):
            yield audio_chunk
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better TTS output.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Basic text cleaning
        cleaned = text.strip()
        
        # Remove excessive whitespace
        cleaned = ' '.join(cleaned.split())
        
        # Ensure sentence ends with punctuation
        if cleaned and not cleaned[-1] in '.!?':
            cleaned += '.'
        
        return cleaned
    
    def get_voice_info(self) -> dict:
        """Get information about the current voice."""
        if self._initialized:
            return self.engine.get_voice_info()
        return {}
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.engine.cleanup()
        self._initialized = False


class GTTSService:
    """High-level GTTS service for easy integration."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.processor = GTTSProcessor(settings)
        
    async def initialize(self) -> None:
        """Initialize the TTS service."""
        await self.processor.initialize()
        
    async def speak(self, text: str) -> np.ndarray:
        """Synthesize text to speech."""
        return await self.processor.process_text(text)
    
    async def speak_stream(self, text_stream: AsyncGenerator[str, None]) -> AsyncGenerator[np.ndarray, None]:
        """Synthesize streaming text to speech."""
        async for audio_chunk in self.processor.process_text_stream(text_stream):
            yield audio_chunk
    
    def get_voice_info(self) -> dict:
        """Get voice information."""
        return self.processor.get_voice_info()
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.processor.cleanup()