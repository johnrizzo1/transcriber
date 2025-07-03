"""
Audio playback module for playing synthesized speech.
"""

import asyncio
import logging
import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from ..config import AudioConfig

logger = logging.getLogger(__name__)


class AudioPlayback:
    """Audio playback handler for TTS output."""
    
    def __init__(self, config: AudioConfig, device_id: Optional[int] = None):
        self.config = config
        self.device_id = device_id or config.output_device
        self.sample_rate = config.sample_rate
        
        # Playback state
        self._playing = False
        self._stop_event = threading.Event()
        self._playback_thread: Optional[threading.Thread] = None
        self._audio_queue: queue.Queue = queue.Queue()
        self._interrupt_flag = threading.Event()  # Flag for interrupting playback
        
        # Get device info
        try:
            if self.device_id is not None:
                self.device_info = sd.query_devices(self.device_id, 'output')
            else:
                self.device_info = sd.query_devices(kind='output')
            logger.info(f"Audio playback device: {self.device_info['name']}")
        except Exception as e:
            logger.warning(f"Could not query audio device: {e}")
            self.device_info = None
    
    async def initialize(self) -> None:
        """Initialize audio playback system."""
        logger.info("Initializing audio playback...")
        
        # Test audio output
        try:
            # Play a short silent test to verify output works
            test_audio = np.zeros(int(self.sample_rate * 0.1), dtype=np.float32)
            sd.play(test_audio, self.sample_rate, device=self.device_id)
            sd.wait()
            logger.info("Audio playback initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize audio playback: {e}")
            raise
    
    async def play(self, audio_data: np.ndarray, sample_rate: Optional[int] = None) -> None:
        """
        Play audio data with interrupt support.
        
        Args:
            audio_data: Audio samples to play
            sample_rate: Sample rate (uses config default if not specified)
        """
        if len(audio_data) == 0:
            return
        
        sample_rate = sample_rate or self.sample_rate
        
        # Ensure audio is the right format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize audio if needed
        max_val = np.abs(audio_data).max()
        if max_val > 1.0:
            audio_data = audio_data / max_val
        
        # Clear interrupt flag
        self._interrupt_flag.clear()
        self._playing = True
        
        # Play with interrupt support
        loop = asyncio.get_running_loop()
        
        def play_interruptible():
            try:
                # Start playback
                sd.play(audio_data, sample_rate, device=self.device_id)
                
                # Calculate expected duration
                duration = len(audio_data) / sample_rate
                checks = int(duration / 0.05) + 10  # Add some buffer
                
                # Wait for completion or interrupt
                for _ in range(checks):
                    if self._interrupt_flag.is_set():
                        # Stop playback immediately
                        sd.stop()
                        break
                    # Check every 50ms
                    threading.Event().wait(0.05)
                    
            except Exception as e:
                logger.error(f"Playback error: {e}")
            finally:
                self._playing = False
        
        await loop.run_in_executor(None, play_interruptible)
    
    async def play_stream(self, audio_stream) -> None:
        """
        Play streaming audio data.
        
        Args:
            audio_stream: Async generator yielding audio chunks
        """
        # Start playback thread if not running
        if not self._playing:
            self._start_playback_thread()
        
        try:
            async for chunk in audio_stream:
                if len(chunk) > 0:
                    self._audio_queue.put(chunk)
            
            # Signal end of stream
            self._audio_queue.put(None)
            
            # Wait for playback to complete
            while not self._audio_queue.empty():
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Stream playback error: {e}")
        finally:
            await self.stop_stream()
    
    def _start_playback_thread(self) -> None:
        """Start the background playback thread."""
        if self._playing:
            return
        
        self._playing = True
        self._stop_event.clear()
        self._playback_thread = threading.Thread(target=self._playback_worker)
        self._playback_thread.daemon = True
        self._playback_thread.start()
    
    def _playback_worker(self) -> None:
        """Background thread for continuous audio playback."""
        stream = None
        
        try:
            # Create output stream
            stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                device=self.device_id,
                blocksize=1024
            )
            stream.start()
            
            # Process audio queue
            while self._playing and not self._stop_event.is_set():
                try:
                    # Get audio chunk from queue
                    chunk = self._audio_queue.get(timeout=0.1)
                    
                    if chunk is None:  # End of stream signal
                        break
                    
                    # Write to stream
                    if len(chunk) > 0:
                        stream.write(chunk)
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Playback worker error: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to create audio stream: {e}")
        finally:
            if stream:
                stream.stop()
                stream.close()
            self._playing = False
    
    async def stop_stream(self) -> None:
        """Stop streaming playback."""
        if self._playing:
            self._stop_event.set()
            if self._playback_thread:
                self._playback_thread.join(timeout=1.0)
            self._playing = False
            
            # Clear queue
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    break
    
    def interrupt(self) -> None:
        """Interrupt current playback immediately."""
        self._interrupt_flag.set()
        
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._playing
        
    async def stop(self) -> None:
        """Stop all playback immediately."""
        self._interrupt_flag.set()
        sd.stop()
        await self.stop_stream()
    
    def get_volume(self) -> float:
        """Get current system volume (0.0 to 1.0)."""
        # Note: sounddevice doesn't provide direct volume control
        # This would need platform-specific implementation
        return 1.0
    
    def set_volume(self, volume: float) -> None:
        """Set system volume (0.0 to 1.0)."""
        # Note: sounddevice doesn't provide direct volume control
        # This would need platform-specific implementation
        logger.warning("Volume control not implemented for sounddevice")
    
    async def cleanup(self) -> None:
        """Clean up audio resources."""
        await self.stop()
        logger.info("Audio playback cleaned up")


class AudioPlayer:
    """Simple audio player for one-shot playback."""
    
    def __init__(self, sample_rate: int = 16000, device_id: Optional[int] = None):
        self.sample_rate = sample_rate
        self.device_id = device_id
    
    def play(self, audio_data: np.ndarray) -> None:
        """Play audio data synchronously."""
        if len(audio_data) == 0:
            return
        
        try:
            sd.play(audio_data, self.sample_rate, device=self.device_id)
            sd.wait()
        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
    
    async def play_async(self, audio_data: np.ndarray) -> None:
        """Play audio data asynchronously."""
        if len(audio_data) == 0:
            return
        
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.play, audio_data)