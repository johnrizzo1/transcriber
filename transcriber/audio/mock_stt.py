"""
Mock STT module for testing voice pipeline without heavy dependencies.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Optional

import numpy as np

from ..config import Settings

logger = logging.getLogger(__name__)


class MockSTTEngine:
    """Mock Speech-to-Text engine for testing."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.mock_responses = [
            "Hello, this is a test message.",
            "How are you doing today?",
            "This is mock speech recognition.",
            "The weather is nice today.",
            "Thank you for testing the system."
        ]
        self.response_index = 0
        
    async def initialize(self) -> None:
        """Initialize the mock STT model."""
        logger.info("Initializing Mock STT engine...")
        await asyncio.sleep(0.1)  # Simulate initialization
        logger.info("Mock STT engine initialized")
        
    async def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Mock transcribe audio data to text.
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Mock transcribed text
        """
        if len(audio_data) == 0:
            return ""
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Calculate a simple metric based on audio
        audio_level = np.mean(np.abs(audio_data))
        
        if audio_level > 0.01:  # If there's significant audio
            # Return a mock response
            response = self.mock_responses[self.response_index % len(self.mock_responses)]
            self.response_index += 1
            
            logger.debug(f"Mock transcribed {len(audio_data)} samples: '{response}'")
            return response
        else:
            logger.debug("No significant audio detected in mock STT")
            return ""
    
    async def transcribe_stream(self, audio_stream: AsyncGenerator[np.ndarray, None]) -> AsyncGenerator[str, None]:
        """
        Mock transcribe streaming audio data.
        
        Args:
            audio_stream: Async generator of audio chunks
            
        Yields:
            Mock transcribed text chunks
        """
        buffer = np.array([], dtype=np.float32)
        min_chunk_duration = 2.0  # seconds
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
                
                # Clear buffer for next chunk
                buffer = np.array([], dtype=np.float32)
        
        # Process remaining buffer
        if len(buffer) > 0:
            text = await self.transcribe_audio(buffer, sample_rate)
            if text:
                yield text
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Mock STT engine cleaned up")


class MockSTTProcessor:
    """Mock STT processor for testing."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine = MockSTTEngine(settings)
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize the mock STT processor."""
        if not self._initialized:
            await self.engine.initialize()
            self._initialized = True
            
    async def process_audio_segments(self, audio_segments: AsyncGenerator[np.ndarray, None]) -> AsyncGenerator[str, None]:
        """
        Process audio segments and yield transcribed text.
        
        Args:
            audio_segments: Async generator of audio segments from VAD
            
        Yields:
            Mock transcribed text
        """
        if not self._initialized:
            raise RuntimeError("Mock STT processor not initialized")
            
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