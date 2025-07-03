"""
Audio recording functionality for debugging and session storage.
"""

import asyncio
import logging
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class AudioRecorder:
    """
    Records audio to WAV files for debugging and playback.
    """
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = False
        self._audio_buffer = []
        self._wav_file = None
        self._file_path = None
        
    def start_recording(self, file_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Start recording audio to a file.
        
        Args:
            file_path: Optional path for the recording. If not provided,
                      generates a timestamped filename.
                      
        Returns:
            Path to the recording file
        """
        if self.recording:
            logger.warning("Already recording, stopping previous recording")
            self.stop_recording()
            
        # Generate filename if not provided
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            recordings_dir = Path("recordings")
            recordings_dir.mkdir(exist_ok=True)
            file_path = recordings_dir / f"audio_{timestamp}.wav"
        else:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
        self._file_path = file_path
        
        # Open WAV file for writing
        self._wav_file = wave.open(str(file_path), 'wb')
        self._wav_file.setnchannels(self.channels)
        self._wav_file.setsampwidth(2)  # 16-bit audio
        self._wav_file.setframerate(self.sample_rate)
        
        self.recording = True
        self._audio_buffer = []
        
        logger.info(f"Started recording to {file_path}")
        return file_path
        
    def add_audio_chunk(self, chunk: np.ndarray) -> None:
        """
        Add an audio chunk to the recording.
        
        Args:
            chunk: Audio data as numpy array (float32)
        """
        if not self.recording:
            return
            
        # Convert float32 to int16
        audio_int16 = (chunk * 32767).astype(np.int16)
        
        # Write to file
        if self._wav_file:
            self._wav_file.writeframes(audio_int16.tobytes())
            
        # Also keep in buffer for potential processing
        self._audio_buffer.append(chunk.copy())
        
    def stop_recording(self) -> Optional[Path]:
        """
        Stop recording and close the file.
        
        Returns:
            Path to the recorded file, or None if not recording
        """
        if not self.recording:
            return None
            
        self.recording = False
        
        if self._wav_file:
            self._wav_file.close()
            self._wav_file = None
            
        file_path = self._file_path
        self._file_path = None
        
        if file_path and file_path.exists():
            file_size = file_path.stat().st_size / 1024  # KB
            duration = len(self._audio_buffer) * len(self._audio_buffer[0]) / self.sample_rate if self._audio_buffer else 0
            logger.info(f"Recording saved: {file_path} ({file_size:.1f}KB, {duration:.1f}s)")
            
        self._audio_buffer = []
        
        return file_path
        
    def get_recording_duration(self) -> float:
        """
        Get the current recording duration in seconds.
        
        Returns:
            Duration in seconds
        """
        if not self._audio_buffer:
            return 0.0
            
        total_samples = sum(len(chunk) for chunk in self._audio_buffer)
        return total_samples / self.sample_rate
        
    async def record_for_duration(self, duration: float, file_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Record audio for a specific duration.
        
        Args:
            duration: Recording duration in seconds
            file_path: Optional path for the recording
            
        Returns:
            Path to the recorded file
        """
        path = self.start_recording(file_path)
        await asyncio.sleep(duration)
        self.stop_recording()
        return path


class SessionRecorder:
    """
    Records entire conversation sessions with metadata.
    """
    
    def __init__(self, session_dir: Path = Path("sessions")):
        self.session_dir = session_dir
        self.session_dir.mkdir(exist_ok=True)
        self.audio_recorder = None
        self.session_id = None
        self.metadata = {}
        
    def start_session(self) -> str:
        """
        Start a new recording session.
        
        Returns:
            Session ID
        """
        # Generate session ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"session_{timestamp}"
        
        # Create session directory
        session_path = self.session_dir / self.session_id
        session_path.mkdir(exist_ok=True)
        
        # Initialize audio recorder
        self.audio_recorder = AudioRecorder()
        audio_path = session_path / "audio.wav"
        self.audio_recorder.start_recording(audio_path)
        
        # Initialize metadata
        self.metadata = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "audio_file": "audio.wav",
            "segments": []
        }
        
        logger.info(f"Started session: {self.session_id}")
        return self.session_id
        
    def add_segment(self, segment_type: str, content: str, start_time: float, end_time: float):
        """
        Add a conversation segment to metadata.
        
        Args:
            segment_type: Type of segment (user, assistant, system)
            content: Text content
            start_time: Start time in seconds from session start
            end_time: End time in seconds from session start
        """
        if not self.metadata:
            return
            
        segment = {
            "type": segment_type,
            "content": content,
            "start_time": start_time,
            "end_time": end_time
        }
        self.metadata["segments"].append(segment)
        
    def stop_session(self) -> Optional[Path]:
        """
        Stop the current session and save metadata.
        
        Returns:
            Path to session directory
        """
        if not self.session_id:
            return None
            
        # Stop audio recording
        if self.audio_recorder:
            self.audio_recorder.stop_recording()
            
        # Update metadata
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["duration"] = self.audio_recorder.get_recording_duration() if self.audio_recorder else 0
        
        # Save metadata
        session_path = self.session_dir / self.session_id
        metadata_path = session_path / "metadata.json"
        
        import json
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
        logger.info(f"Session saved: {session_path}")
        
        # Reset
        self.session_id = None
        self.audio_recorder = None
        self.metadata = {}
        
        return session_path