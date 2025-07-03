"""
Audio capture functionality.
"""

import asyncio
import logging
from collections import deque
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd

from ..config import AudioConfig
from .recorder import AudioRecorder

logger = logging.getLogger(__name__)


class AudioCapture:
    """
    Async audio capture using sounddevice.
    """
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.is_recording = False
        self._stream = None
        self._audio_queue = asyncio.Queue()
        self._buffer = deque(maxlen=100)  # Keep last 10 seconds at 16kHz
        self._recorder = None  # Optional AudioRecorder for debugging
        self._muted = False  # Mute flag to prevent feedback
        
    async def start_capture(self) -> None:
        """Start audio capture."""
        if self.is_recording:
            logger.warning("Audio capture already started")
            return
            
        try:
            logger.info(f"Starting audio capture on device {self.config.input_device}")
            
            # Calculate chunk size
            chunk_size = int(self.config.sample_rate * self.config.chunk_duration)
            
            def audio_callback(indata, frames, time, status):
                """Callback for audio input."""
                if status:
                    logger.warning(f"Audio callback status: {status}")
                
                # If muted, send silence instead of real audio
                if self._muted:
                    audio_data = np.zeros(indata.shape[0], dtype=np.float32)
                else:
                    # Convert to mono if needed
                    if indata.shape[1] > 1:
                        audio_data = np.mean(indata, axis=1)
                    else:
                        audio_data = indata[:, 0]
                
                # Add to buffer and queue
                self._buffer.append(audio_data.copy())
                
                # Record to file if recorder is active (record real audio, not muted)
                if self._recorder and not self._muted:
                    real_audio = np.mean(indata, axis=1) if indata.shape[1] > 1 else indata[:, 0]
                    self._recorder.add_audio_chunk(real_audio)
                
                # Put in queue for async processing
                try:
                    self._audio_queue.put_nowait(audio_data.copy())
                except asyncio.QueueFull:
                    logger.warning("Audio queue full, dropping frame")
            
            # Start the stream
            self._stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                device=self.config.input_device,
                blocksize=chunk_size,
                callback=audio_callback,
                dtype=np.float32
            )
            
            self._stream.start()
            self.is_recording = True
            logger.info("Audio capture started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            raise
    
    async def stop_capture(self) -> None:
        """Stop audio capture."""
        if not self.is_recording:
            return
            
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
                
            self.is_recording = False
            logger.info("Audio capture stopped")
            
        except Exception as e:
            logger.error(f"Error stopping audio capture: {e}")
    
    async def get_audio_chunks(self) -> AsyncGenerator[np.ndarray, None]:
        """
        Async generator yielding audio chunks.
        
        Yields:
            Audio chunks as numpy arrays
        """
        while self.is_recording:
            try:
                # Wait for audio data with timeout
                chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=1.0)
                yield chunk
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error getting audio chunk: {e}")
                break
    
    def get_recent_audio(self, duration: float) -> Optional[np.ndarray]:
        """
        Get recent audio from buffer.
        
        Args:
            duration: Duration in seconds
            
        Returns:
            Audio data or None if not enough data
        """
        if not self._buffer:
            return None
            
        # Calculate how many chunks we need
        chunk_duration = self.config.chunk_duration
        needed_chunks = int(duration / chunk_duration)
        
        if len(self._buffer) < needed_chunks:
            # Return all available data
            chunks = list(self._buffer)
        else:
            # Return the last N chunks
            chunks = list(self._buffer)[-needed_chunks:]
        
        if not chunks:
            return None
            
        # Concatenate chunks
        return np.concatenate(chunks)
    
    def get_audio_level(self) -> float:
        """
        Get current audio level (RMS).
        
        Returns:
            Audio level between 0 and 1
        """
        if not self._buffer:
            return 0.0
            
        # Get the last chunk
        last_chunk = self._buffer[-1]
        rms = np.sqrt(np.mean(last_chunk**2))
        return min(rms * 10, 1.0)  # Scale and clamp to [0, 1]
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_capture()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_capture()
        
    def start_recording(self, file_path: Optional[str] = None) -> Optional[Path]:
        """
        Start recording audio to file.
        
        Args:
            file_path: Optional path for the recording
            
        Returns:
            Path to the recording file
        """
        if not self._recorder:
            self._recorder = AudioRecorder(
                sample_rate=self.config.sample_rate,
                channels=self.config.channels
            )
        
        return self._recorder.start_recording(file_path)
        
    def stop_recording(self) -> Optional[Path]:
        """
        Stop recording audio to file.
        
        Returns:
            Path to the recorded file
        """
        if self._recorder:
            path = self._recorder.stop_recording()
            self._recorder = None
            return path
        return None
    
    def mute(self) -> None:
        """Mute audio input to prevent feedback."""
        self._muted = True
    
    def unmute(self) -> None:
        """Unmute audio input."""
        self._muted = False
    
    def is_muted(self) -> bool:
        """Check if audio is muted."""
        return self._muted
    
