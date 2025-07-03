"""
Speech-to-Text module using faster-whisper for real-time transcription.
"""

import asyncio
import logging
import tempfile
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

from ..config import Settings

logger = logging.getLogger(__name__)


class STTEngine:
    """Speech-to-Text engine using faster-whisper."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model: Optional[WhisperModel] = None
        self.temp_dir = Path(tempfile.gettempdir()) / "transcriber_stt"
        self.temp_dir.mkdir(exist_ok=True)
        
    async def initialize(self) -> None:
        """Initialize the STT model."""
        if WhisperModel is None:
            raise ImportError("faster-whisper not available. Install with: pip install faster-whisper")
        
        logger.info("Initializing STT engine...")
        
        # Run model loading in thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        
        def load_model():
            return WhisperModel(
                model_size_or_path=self.settings.whisper.model,
                device=self.settings.whisper.device,
                compute_type=self.settings.whisper.compute_type,
                download_root=str(self.temp_dir / "models"),
                local_files_only=False,
            )
        
        self.model = await loop.run_in_executor(None, load_model)
        logger.info(f"STT engine initialized with model: {self.settings.whisper.model}")
        
    async def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Transcribed text
        """
        if self.model is None:
            raise RuntimeError("STT engine not initialized")
            
        if len(audio_data) == 0:
            return ""
            
        # Normalize audio to [-1, 1] range
        audio_normalized = audio_data.astype(np.float32)
        if audio_normalized.max() > 1.0 or audio_normalized.min() < -1.0:
            audio_normalized = audio_normalized / np.max(np.abs(audio_normalized))
        
        start_time = time.time()
        
        # Run transcription in thread pool
        loop = asyncio.get_running_loop()
        
        def transcribe():
            segments, info = self.model.transcribe(
                audio_normalized,
                language=self.settings.whisper.language,
                beam_size=self.settings.whisper.beam_size,
                best_of=self.settings.whisper.best_of,
                temperature=self.settings.whisper.temperature,
                compression_ratio_threshold=self.settings.whisper.compression_ratio_threshold,
                log_prob_threshold=self.settings.whisper.log_prob_threshold,
                no_speech_threshold=self.settings.whisper.no_speech_threshold,
                condition_on_previous_text=self.settings.whisper.condition_on_previous_text,
                initial_prompt=self.settings.whisper.initial_prompt,
                word_timestamps=False,
                vad_filter=True,
                vad_parameters=dict(
                    onset=0.500,
                    offset=0.363
                )
            )
            
            # Combine all segments into a single text
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())
            
            return " ".join(text_parts).strip()
        
        try:
            text = await loop.run_in_executor(None, transcribe)
            elapsed_time = time.time() - start_time
            
            if text:
                logger.debug(f"Transcribed {len(audio_data)} samples in {elapsed_time:.2f}s: '{text[:100]}...'")
            else:
                logger.debug(f"No speech detected in {len(audio_data)} samples ({elapsed_time:.2f}s)")
                
            return text
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
    
    async def transcribe_stream(self, audio_stream: AsyncGenerator[np.ndarray, None]) -> AsyncGenerator[str, None]:
        """
        Transcribe streaming audio data.
        
        Args:
            audio_stream: Async generator of audio chunks
            
        Yields:
            Transcribed text chunks
        """
        buffer = np.array([], dtype=np.float32)
        min_chunk_duration = self.settings.whisper.chunk_duration
        sample_rate = 16000
        min_samples = int(min_chunk_duration * sample_rate)
        
        async for audio_chunk in audio_stream:
            if len(audio_chunk) == 0:
                continue
                
            # Add to buffer
            buffer = np.concatenate([buffer, audio_chunk.astype(np.float32)])
            
            # Process when we have enough samples
            if len(buffer) >= min_samples:
                text = await self.transcribe_audio(buffer, sample_rate)
                if text:
                    yield text
                
                # Keep a small overlap for better continuity
                overlap_samples = int(0.2 * sample_rate)  # 200ms overlap
                buffer = buffer[-overlap_samples:] if len(buffer) > overlap_samples else np.array([], dtype=np.float32)
        
        # Process remaining buffer
        if len(buffer) > 0:
            text = await self.transcribe_audio(buffer, sample_rate)
            if text:
                yield text
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.model is not None:
            # faster-whisper doesn't have explicit cleanup
            self.model = None
            logger.info("STT engine cleaned up")


class STTProcessor:
    """High-level STT processor that handles audio preprocessing and transcription."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine = STTEngine(settings)
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize the STT processor."""
        if not self._initialized:
            await self.engine.initialize()
            self._initialized = True
            
    async def process_audio_segments(self, audio_segments: AsyncGenerator[np.ndarray, None]) -> AsyncGenerator[str, None]:
        """
        Process audio segments and yield transcribed text.
        
        Args:
            audio_segments: Async generator of audio segments from VAD
            
        Yields:
            Transcribed text
        """
        if not self._initialized:
            raise RuntimeError("STT processor not initialized")
            
        async for segment in audio_segments:
            if len(segment) == 0:
                continue
                
            text = await self.engine.transcribe_audio(segment)
            if text:
                yield text
                
    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.engine.cleanup()
        self._initialized = False