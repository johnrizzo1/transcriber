"""
Voice Activity Detection (VAD) for speech segmentation.
"""

import asyncio
import logging
from collections import deque
from collections.abc import AsyncGenerator
from typing import Optional

import numpy as np
import webrtcvad

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """
    Detects voice activity in audio streams using WebRTC VAD.
    """
    
    def __init__(
        self, 
        sample_rate: int = 16000,
        frame_duration: int = 30,  # milliseconds (10, 20, or 30)
        aggressiveness: int = 3,   # 0-3, higher = more aggressive filtering
        speech_threshold: float = 0.5,  # Fraction of frames that must be speech
        silence_duration: float = 1.0   # Seconds of silence to end speech
    ):
        """
        Initialize VAD.
        
        Args:
            sample_rate: Audio sample rate (8000, 16000, 32000, or 48000)
            frame_duration: Frame duration in ms (10, 20, or 30)
            aggressiveness: VAD aggressiveness (0-3)
            speech_threshold: Fraction of frames in window that must be speech
            silence_duration: Seconds of silence before ending speech segment
        """
        # Validate parameters
        if sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError(f"Sample rate {sample_rate} not supported by WebRTC VAD")
        if frame_duration not in [10, 20, 30]:
            raise ValueError(f"Frame duration {frame_duration}ms not supported")
            
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration / 1000)
        self.speech_threshold = speech_threshold
        self.silence_duration = silence_duration
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(aggressiveness)
        
        # State tracking
        self.is_speaking = False
        self.speech_frames = deque()
        self.silence_frames = 0
        self.speech_start_time = None
        
        # Ring buffer for smoothing decisions
        self.frame_window_size = 10  # Number of frames to consider
        self.frame_decisions = deque(maxlen=self.frame_window_size)
        
        logger.info(f"VAD initialized: {sample_rate}Hz, {frame_duration}ms frames, aggressiveness={aggressiveness}")
        
    def process_audio(self, audio_data: np.ndarray) -> tuple[bool, Optional[np.ndarray]]:
        """
        Process audio data and detect speech segments.
        
        Args:
            audio_data: Audio data as numpy array (float32)
            
        Returns:
            Tuple of (is_speech_complete, speech_segment)
            - is_speech_complete: True if a complete speech segment was detected
            - speech_segment: The complete speech audio if detected, None otherwise
        """
        # Convert float32 to int16 for VAD
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Process in frames
        speech_detected = False
        frames_processed = 0
        
        for i in range(0, len(audio_int16), self.frame_size):
            frame = audio_int16[i:i + self.frame_size]
            
            # Skip incomplete frames
            if len(frame) < self.frame_size:
                continue
                
            # Check if frame contains speech
            try:
                is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
                self.frame_decisions.append(is_speech)
                frames_processed += 1
                
                # Calculate speech ratio in window
                if len(self.frame_decisions) > 0:
                    speech_ratio = sum(self.frame_decisions) / len(self.frame_decisions)
                    speech_detected = speech_ratio >= self.speech_threshold
                    
            except Exception as e:
                logger.error(f"VAD error: {e}")
                continue
        
        # Update state based on speech detection
        return self._update_state(speech_detected, audio_data)
        
    def _update_state(self, speech_detected: bool, audio_data: np.ndarray) -> tuple[bool, Optional[np.ndarray]]:
        """
        Update VAD state based on speech detection.
        
        Returns:
            Tuple of (is_speech_complete, speech_segment)
        """
        if speech_detected:
            if not self.is_speaking:
                # Start of speech
                self.is_speaking = True
                self.speech_start_time = len(self.speech_frames) * len(audio_data) / self.sample_rate
                logger.debug("Speech started")
                
            # Add to speech buffer
            self.speech_frames.append(audio_data.copy())
            self.silence_frames = 0
            
        else:  # No speech detected
            if self.is_speaking:
                # Add to buffer during silence
                self.speech_frames.append(audio_data.copy())
                self.silence_frames += 1
                
                # Check if silence duration exceeded
                silence_duration = self.silence_frames * len(audio_data) / self.sample_rate
                
                if silence_duration >= self.silence_duration:
                    # End of speech segment
                    self.is_speaking = False
                    
                    # Combine all frames into speech segment
                    if self.speech_frames:
                        speech_segment = np.concatenate(list(self.speech_frames))
                        duration = len(speech_segment) / self.sample_rate
                        logger.info(f"Speech segment complete: {duration:.1f}s")
                        
                        # Reset state
                        self.speech_frames.clear()
                        self.silence_frames = 0
                        self.frame_decisions.clear()
                        
                        return True, speech_segment
                        
        return False, None
        
    def reset(self):
        """Reset VAD state."""
        self.is_speaking = False
        self.speech_frames.clear()
        self.silence_frames = 0
        self.frame_decisions.clear()
        self.speech_start_time = None
        logger.debug("VAD state reset")
        
    def get_state(self) -> dict:
        """Get current VAD state for debugging."""
        return {
            "is_speaking": self.is_speaking,
            "buffered_frames": len(self.speech_frames),
            "silence_frames": self.silence_frames,
            "recent_decisions": list(self.frame_decisions)
        }


class SpeechSegmenter:
    """
    Segments continuous audio into speech segments using VAD.
    """
    
    def __init__(self, vad_config: dict = None):
        """
        Initialize speech segmenter.
        
        Args:
            vad_config: Configuration for VAD
        """
        vad_config = vad_config or {}
        self.vad = VoiceActivityDetector(**vad_config)
        self.segments = []
        self._audio_buffer = []
        self._start_time = 0
        
    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[dict]:
        """
        Process an audio chunk and return completed speech segment if any.
        
        Args:
            audio_chunk: Audio data chunk
            
        Returns:
            Speech segment dict if complete, None otherwise
        """
        # Add to buffer for position tracking
        self._audio_buffer.append(audio_chunk)
        
        # Process with VAD
        is_complete, speech_audio = self.vad.process_audio(audio_chunk)
        
        if is_complete and speech_audio is not None:
            # Calculate timing
            end_time = sum(len(chunk) for chunk in self._audio_buffer) / self.vad.sample_rate
            duration = len(speech_audio) / self.vad.sample_rate
            start_time = end_time - duration
            
            segment = {
                "audio": speech_audio,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "sample_rate": self.vad.sample_rate
            }
            
            self.segments.append(segment)
            return segment
            
        return None
        
    def get_segments(self) -> list[dict]:
        """Get all detected speech segments."""
        return self.segments.copy()
        
    def reset(self):
        """Reset segmenter state."""
        self.vad.reset()
        self.segments.clear()
        self._audio_buffer.clear()
        self._start_time = 0


class VADProcessor:
    """
    Async VAD processor for streaming audio processing.
    """
    
    def __init__(self, vad_threshold: float = 0.5):
        """
        Initialize VAD processor.
        
        Args:
            vad_threshold: Speech detection threshold
        """
        self.vad_threshold = vad_threshold
        self.segmenter = SpeechSegmenter({
            'speech_threshold': vad_threshold,
            'silence_duration': 1.0
        })
        
    async def process_audio_stream(
        self, 
        audio_stream: AsyncGenerator[np.ndarray, None]
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Process streaming audio and yield speech segments.
        
        Args:
            audio_stream: Async generator of audio chunks
            
        Yields:
            Complete speech segments
        """
        async for audio_chunk in audio_stream:
            segment = self.segmenter.process_chunk(audio_chunk)
            if segment is not None:
                yield segment['audio']
    
    def reset(self):
        """Reset processor state."""
        self.segmenter.reset()