"""
Text-to-Speech module using Piper TTS for voice synthesis.
"""

import asyncio
import io
import logging
import wave
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from piper import PiperVoice
    from piper.download import ensure_voice_exists
except ImportError:
    PiperVoice = None
    ensure_voice_exists = None

from ..config import Settings

logger = logging.getLogger(__name__)


class TTSEngine:
    """Text-to-Speech engine using Piper."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.voice: Optional[PiperVoice] = None
        self.models_dir = Path(self.settings.piper.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    async def initialize(self) -> None:
        """Initialize the TTS engine."""
        if PiperVoice is None or ensure_voice_exists is None:
            raise ImportError("piper-tts not available. Install with: pip install piper-tts")
        
        logger.info("Initializing TTS engine...")
        
        # Run model loading in thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        
        def load_voice():
            # Ensure voice model exists
            ensure_voice_exists(
                self.settings.piper.model,
                str(self.models_dir),
                download_dir=str(self.models_dir)
            )
            
            # Load the voice
            model_path = self.models_dir / f"{self.settings.piper.model}.onnx"
            config_path = self.models_dir / f"{self.settings.piper.model}.onnx.json"
            
            if not model_path.exists() or not config_path.exists():
                raise FileNotFoundError(f"Voice model files not found: {model_path}")
            
            return PiperVoice.load(str(model_path), str(config_path))
        
        try:
            self.voice = await loop.run_in_executor(None, load_voice)
            logger.info(f"TTS engine initialized with voice: {self.settings.piper.model}")
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            raise
        
    async def synthesize_text(self, text: str) -> np.ndarray:
        """
        Synthesize text to audio.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio data as numpy array
        """
        if self.voice is None:
            raise RuntimeError("TTS engine not initialized")
            
        if not text.strip():
            return np.array([], dtype=np.float32)
        
        # Run synthesis in thread pool
        loop = asyncio.get_running_loop()
        
        def synthesize():
            # Use BytesIO to capture audio data
            audio_buffer = io.BytesIO()
            
            # Synthesize audio
            self.voice.synthesize(
                text,
                audio_buffer,
                sentence_silence=self.settings.piper.sentence_silence
            )
            
            # Get audio data
            audio_buffer.seek(0)
            
            # Read WAV data
            with wave.open(audio_buffer, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16)
                
                # Convert to float32 and normalize
                audio_float = audio_data.astype(np.float32) / 32768.0
                
                # Apply volume adjustment
                audio_float *= self.settings.piper.volume
                
                return audio_float
        
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
        if self.voice is None:
            return {}
        
        return {
            "model": self.settings.piper.model,
            "sample_rate": self.voice.config.sample_rate,
            "language": getattr(self.voice.config, 'language', 'unknown'),
            "speaker": getattr(self.voice.config, 'speaker', 'default')
        }
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.voice is not None:
            # Piper doesn't have explicit cleanup
            self.voice = None
            logger.info("TTS engine cleaned up")


class TTSProcessor:
    """High-level TTS processor that handles text preprocessing and audio output."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine = TTSEngine(settings)
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
            raise RuntimeError("TTS processor not initialized")
        
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
            raise RuntimeError("TTS processor not initialized")
        
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
        if cleaned and cleaned[-1] not in '.!?':
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


class TTSService:
    """High-level TTS service for easy integration."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.processor = TTSProcessor(settings)
        
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