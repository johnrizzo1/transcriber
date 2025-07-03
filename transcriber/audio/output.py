"""
Audio output and playback functionality.
"""

import asyncio
import logging
import threading
from collections.abc import AsyncGenerator
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioOutput:
    """
    Manages audio playback with queuing and interruption support.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device: Optional[int] = None,
        buffer_size: int = 50  # Number of chunks to buffer
    ):
        """
        Initialize audio output.
        
        Args:
            sample_rate: Sample rate for playback
            channels: Number of audio channels
            device: Output device index (None for default)
            buffer_size: Maximum number of chunks to buffer
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.buffer_size = buffer_size
        
        # Playback state
        self.is_playing = False
        self._stream = None
        self._audio_queue = asyncio.Queue(maxsize=buffer_size)
        self._playback_thread = None
        self._stop_event = threading.Event()
        
        # Statistics
        self.chunks_played = 0
        self.chunks_dropped = 0
        
        logger.info(f"Audio output initialized: {sample_rate}Hz, device={device}")
        
    async def start(self):
        """Start audio output system."""
        if self.is_playing:
            logger.warning("Audio output already started")
            return
            
        self.is_playing = True
        self._stop_event.clear()
        
        # Start playback in thread
        self._playback_thread = threading.Thread(target=self._playback_worker)
        self._playback_thread.start()
        
        logger.info("Audio output started")
        
    async def stop(self):
        """Stop audio output system."""
        if not self.is_playing:
            return
            
        self.is_playing = False
        self._stop_event.set()
        
        # Clear queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
                
        # Wait for thread to finish
        if self._playback_thread:
            self._playback_thread.join(timeout=1.0)
            
        logger.info(f"Audio output stopped. Played: {self.chunks_played}, Dropped: {self.chunks_dropped}")
        
    async def play_audio(self, audio_data: np.ndarray, block: bool = False):
        """
        Play audio data.
        
        Args:
            audio_data: Audio data to play
            block: Whether to wait for playback to complete
        """
        if not self.is_playing:
            await self.start()
            
        try:
            if block:
                await self._audio_queue.put(audio_data)
            else:
                self._audio_queue.put_nowait(audio_data)
        except asyncio.QueueFull:
            self.chunks_dropped += 1
            logger.warning("Audio output queue full, dropping audio")
            
    async def play_stream(self, audio_stream: AsyncGenerator[np.ndarray, None]):
        """
        Play audio from an async generator stream.
        
        Args:
            audio_stream: Async generator yielding audio chunks
        """
        if not self.is_playing:
            await self.start()
            
        async for chunk in audio_stream:
            await self.play_audio(chunk, block=True)
            
    def clear_queue(self):
        """Clear all pending audio from the queue."""
        cleared = 0
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
                cleared += 1
            except asyncio.QueueEmpty:
                break
                
        if cleared > 0:
            logger.info(f"Cleared {cleared} audio chunks from queue")
            
    def _playback_worker(self):
        """Worker thread for audio playback."""
        try:
            # Open output stream
            stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                device=self.device,
                dtype=np.float32
            )
            stream.start()
            
            logger.debug("Playback stream started")
            
            # Playback loop
            while not self._stop_event.is_set():
                try:
                    # Get audio from queue (with timeout)
                    audio_data = asyncio.run_coroutine_threadsafe(
                        asyncio.wait_for(self._audio_queue.get(), timeout=0.1),
                        asyncio.get_event_loop()
                    ).result()
                    
                    # Play audio
                    stream.write(audio_data)
                    self.chunks_played += 1
                    
                except (asyncio.TimeoutError, concurrent.futures.TimeoutError):
                    continue
                except Exception as e:
                    logger.error(f"Playback error: {e}")
                    
            # Clean up
            stream.stop()
            stream.close()
            
        except Exception as e:
            logger.error(f"Playback worker error: {e}")
            
        logger.debug("Playback worker stopped")
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


class AudioMixer:
    """
    Mixes multiple audio streams for simultaneous playback.
    """
    
    def __init__(self, output: AudioOutput):
        """
        Initialize audio mixer.
        
        Args:
            output: AudioOutput instance to use for playback
        """
        self.output = output
        self.active_streams = {}
        self.mix_buffer = None
        
    async def add_stream(self, stream_id: str, audio_stream: AsyncGenerator[np.ndarray, None]):
        """
        Add an audio stream to the mixer.
        
        Args:
            stream_id: Unique identifier for the stream
            audio_stream: Async generator yielding audio chunks
        """
        self.active_streams[stream_id] = audio_stream
        
        # Process stream
        try:
            async for chunk in audio_stream:
                # For now, just play directly (no mixing)
                # TODO: Implement proper mixing
                await self.output.play_audio(chunk)
        finally:
            # Remove stream when done
            self.active_streams.pop(stream_id, None)
            
    def remove_stream(self, stream_id: str):
        """Remove a stream from the mixer."""
        self.active_streams.pop(stream_id, None)
        
    def clear_all(self):
        """Clear all active streams."""
        self.active_streams.clear()
        self.output.clear_queue()


# Import concurrent.futures for the playback worker
import concurrent.futures