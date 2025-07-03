"""
Whisper-based Speech-to-Text processor using faster-whisper.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import AsyncGenerator, Optional, Union
from collections.abc import AsyncGenerator as AsyncGeneratorType

import numpy as np

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    WhisperModel = None

from ..config import Settings

logger = logging.getLogger(__name__)


class WhisperSTTProcessor:
    """Speech-to-Text processor using Whisper."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
        # Model settings
        self.model_size = getattr(settings.whisper, 'model', 'base')
        self.device = getattr(settings.whisper, 'device', 'cpu')
        self.compute_type = getattr(settings.whisper, 'compute_type', 'int8')
        self.language = getattr(settings.whisper, 'language', 'en')
        
        # Processing settings
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.min_speech_duration = 0.5  # Minimum speech duration in seconds
        self.max_speech_duration = 30.0  # Maximum speech duration in seconds
        
        self.model: Optional[WhisperModel] = None
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize the Whisper model."""
        if not WHISPER_AVAILABLE:
            raise RuntimeError("faster-whisper is not installed. Run: pip install faster-whisper")
        
        if self._initialized:
            return
        
        logger.info(f"Initializing Whisper model: {self.model_size} on {self.device}")
        
        # Initialize in thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        
        def load_model():
            try:
                # Download and load model
                model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=str(Path.home() / ".cache" / "whisper")
                )
                return model
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise
        
        try:
            start_time = time.time()
            self.model = await loop.run_in_executor(None, load_model)
            load_time = time.time() - start_time
            
            logger.info(f"Whisper model loaded successfully in {load_time:.2f}s")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize Whisper: {e}")
            raise
    
    async def transcribe_audio(self, audio_data: np.ndarray) -> str:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Audio samples as numpy array
            
        Returns:
            Transcribed text
        """
        if not self._initialized:
            raise RuntimeError("Whisper processor not initialized")
        
        if len(audio_data) == 0:
            return ""
        
        # Ensure audio is float32 and normalized
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize to [-1, 1] range if needed
        if np.abs(audio_data).max() > 1.0:
            audio_data = audio_data / np.abs(audio_data).max()
        
        # Run transcription in thread pool
        loop = asyncio.get_running_loop()
        
        def transcribe():
            try:
                segments, info = self.model.transcribe(
                    audio_data,
                    language=self.language,
                    task="transcribe",
                    beam_size=5,
                    best_of=5,
                    patience=1.0,
                    length_penalty=1.0,
                    temperature=0.0,
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=True,
                    initial_prompt=None,
                    word_timestamps=False,
                    vad_filter=True,
                    vad_parameters={
                        "threshold": 0.5,
                        "min_speech_duration_ms": 250,
                        "max_speech_duration_s": float('inf'),
                        "min_silence_duration_ms": 2000,
                        "speech_pad_ms": 400
                    }
                )
                
                # Collect all segments
                text_parts = []
                for segment in segments:
                    text_parts.append(segment.text.strip())
                
                return " ".join(text_parts)
                
            except Exception as e:
                logger.error(f"Transcription error: {e}")
                return ""
        
        try:
            start_time = time.time()
            text = await loop.run_in_executor(None, transcribe)
            transcribe_time = time.time() - start_time
            
            if text:
                logger.debug(f"Transcribed in {transcribe_time:.2f}s: {text}")
            
            return text
            
        except Exception as e:
            logger.error(f"Failed to transcribe audio: {e}")
            return ""
    
    async def process_audio_segments(
        self, 
        segments: AsyncGeneratorType[np.ndarray, None]
    ) -> AsyncGenerator[str, None]:
        """
        Process audio segments and yield transcriptions.
        
        Args:
            segments: Async generator of audio segments
            
        Yields:
            Transcribed text for each segment
        """
        if not self._initialized:
            await self.initialize()
        
        # Buffer to accumulate audio
        audio_buffer = []
        buffer_duration = 0.0
        
        async for segment in segments:
            if len(segment) == 0:
                continue
            
            # Add to buffer
            audio_buffer.append(segment)
            buffer_duration += len(segment) / self.sample_rate
            
            # Process when we have enough audio or max duration reached
            if buffer_duration >= self.min_speech_duration:
                # Concatenate buffer
                audio_data = np.concatenate(audio_buffer)
                
                # Transcribe
                text = await self.transcribe_audio(audio_data)
                
                if text:
                    yield text
                
                # Reset buffer
                audio_buffer = []
                buffer_duration = 0.0
        
        # Process any remaining audio
        if audio_buffer and buffer_duration > 0.1:  # At least 100ms
            audio_data = np.concatenate(audio_buffer)
            text = await self.transcribe_audio(audio_data)
            if text:
                yield text
    
    async def transcribe_stream(
        self,
        audio_stream: AsyncGeneratorType[np.ndarray, None]
    ) -> AsyncGenerator[str, None]:
        """
        Transcribe streaming audio with real-time processing.
        
        Args:
            audio_stream: Async generator of audio chunks
            
        Yields:
            Transcribed text chunks
        """
        if not self._initialized:
            await self.initialize()
        
        # Process with sliding window for better real-time performance
        window_size = int(self.sample_rate * 2.0)  # 2 second window
        step_size = int(self.sample_rate * 1.0)    # 1 second step
        
        audio_window = np.array([], dtype=np.float32)
        last_transcription = ""
        
        async for chunk in audio_stream:
            if len(chunk) == 0:
                continue
            
            # Add to window
            audio_window = np.concatenate([audio_window, chunk])
            
            # Process when window is full
            if len(audio_window) >= window_size:
                # Get window for transcription
                process_audio = audio_window[:window_size]
                
                # Transcribe
                text = await self.transcribe_audio(process_audio)
                
                # Only yield if different from last
                if text and text != last_transcription:
                    yield text
                    last_transcription = text
                
                # Slide window
                audio_window = audio_window[step_size:]
        
        # Process remaining audio
        if len(audio_window) > self.sample_rate * 0.5:  # At least 0.5 seconds
            text = await self.transcribe_audio(audio_window)
            if text and text != last_transcription:
                yield text
    
    def get_info(self) -> dict:
        """Get information about the STT processor."""
        return {
            "engine": "Whisper (faster-whisper)",
            "model": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "language": self.language,
            "sample_rate": self.sample_rate,
            "initialized": self._initialized
        }
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.model:
            # Model cleanup if needed
            self.model = None
        self._initialized = False
        logger.info("Whisper STT processor cleaned up")


class WhisperSTTService:
    """High-level Whisper STT service for easy integration."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.processor = WhisperSTTProcessor(settings)
        
    async def initialize(self) -> None:
        """Initialize the STT service."""
        await self.processor.initialize()
        
    async def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio data to text."""
        return await self.processor.transcribe_audio(audio_data)
    
    async def transcribe_segments(
        self, 
        segments: AsyncGeneratorType[np.ndarray, None]
    ) -> AsyncGenerator[str, None]:
        """Transcribe audio segments."""
        async for text in self.processor.process_audio_segments(segments):
            yield text
    
    async def transcribe_stream(
        self,
        audio_stream: AsyncGeneratorType[np.ndarray, None]
    ) -> AsyncGenerator[str, None]:
        """Transcribe streaming audio."""
        async for text in self.processor.transcribe_stream(audio_stream):
            yield text
    
    def get_info(self) -> dict:
        """Get STT service information."""
        return self.processor.get_info()
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.processor.cleanup()


# Fallback for when Whisper is not available
class MockWhisperSTT:
    """Mock Whisper STT for when faster-whisper is not installed."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        logger.warning("Using mock Whisper STT - install faster-whisper for real transcription")
        
    async def initialize(self) -> None:
        """Initialize mock STT."""
        pass
        
    async def transcribe(self, audio_data: np.ndarray) -> str:
        """Mock transcription."""
        duration = len(audio_data) / 16000
        return f"[Mock transcription of {duration:.1f}s audio]"
    
    async def transcribe_segments(
        self, 
        segments: AsyncGeneratorType[np.ndarray, None]
    ) -> AsyncGenerator[str, None]:
        """Mock segment transcription."""
        count = 0
        async for segment in segments:
            count += 1
            yield f"[Mock segment {count}]"
    
    async def transcribe_stream(
        self,
        audio_stream: AsyncGeneratorType[np.ndarray, None]
    ) -> AsyncGenerator[str, None]:
        """Mock stream transcription."""
        count = 0
        async for chunk in audio_stream:
            count += 1
            if count % 10 == 0:  # Every 10 chunks
                yield f"[Mock stream transcription {count}]"
    
    def get_info(self) -> dict:
        """Get mock info."""
        return {
            "engine": "Mock Whisper",
            "warning": "Install faster-whisper for real transcription"
        }
    
    async def cleanup(self) -> None:
        """Cleanup mock."""
        pass


# Export the appropriate service based on availability
if WHISPER_AVAILABLE:
    DefaultWhisperService = WhisperSTTService
else:
    DefaultWhisperService = MockWhisperSTT