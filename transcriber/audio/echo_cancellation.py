"""
Simple echo cancellation and audio filtering.
"""

import numpy as np
from typing import Deque
from collections import deque


class SimpleEchoCancellation:
    """Simple echo cancellation using audio level analysis."""
    
    def __init__(self, buffer_size: int = 100):
        self.buffer_size = buffer_size
        self.recent_levels: Deque[float] = deque(maxlen=buffer_size)
        self.baseline_noise = 0.005  # Baseline noise level
        self.speaking_threshold_multiplier = 3.0  # Must be 3x above baseline
        
    def add_audio_level(self, level: float) -> None:
        """Add audio level to recent history."""
        self.recent_levels.append(level)
        
        # Update baseline noise level (running average of low levels)
        if level < self.baseline_noise * 2:
            self.baseline_noise = self.baseline_noise * 0.95 + level * 0.05
    
    def is_likely_speech(self, current_level: float) -> bool:
        """
        Determine if current audio level likely represents human speech.
        
        Args:
            current_level: Current audio RMS level
            
        Returns:
            True if likely human speech, False if likely echo/noise
        """
        # Must be significantly above baseline
        threshold = max(self.baseline_noise * self.speaking_threshold_multiplier, 0.015)
        
        if current_level < threshold:
            return False
        
        # Check for sudden spikes (typical of echo/feedback)
        if len(self.recent_levels) >= 3:
            recent_avg = sum(list(self.recent_levels)[-3:]) / 3
            if current_level > recent_avg * 5:  # Sudden 5x spike
                return False
        
        return True
    
    def get_dynamic_threshold(self) -> float:
        """Get current dynamic speech threshold."""
        return max(self.baseline_noise * self.speaking_threshold_multiplier, 0.015)


class AudioLevelFilter:
    """Filter for managing audio levels and detecting speech patterns."""
    
    def __init__(self):
        self.echo_canceller = SimpleEchoCancellation()
        self.consecutive_speech_chunks = 0
        self.consecutive_silence_chunks = 0
        
    def process_chunk(self, audio_chunk: np.ndarray) -> dict:
        """
        Process audio chunk and return analysis.
        
        Returns:
            dict with 'level', 'is_speech', 'is_likely_human', 'threshold'
        """
        # Calculate RMS level
        level = np.sqrt(np.mean(audio_chunk**2))
        
        # Update echo cancellation
        self.echo_canceller.add_audio_level(level)
        
        # Determine if this is likely human speech
        is_likely_human = self.echo_canceller.is_likely_speech(level)
        
        # Basic speech detection
        threshold = self.echo_canceller.get_dynamic_threshold()
        is_speech = level > threshold
        
        # Update counters
        if is_speech and is_likely_human:
            self.consecutive_speech_chunks += 1
            self.consecutive_silence_chunks = 0
        else:
            self.consecutive_speech_chunks = 0
            self.consecutive_silence_chunks += 1
        
        return {
            'level': level,
            'is_speech': is_speech,
            'is_likely_human': is_likely_human,
            'threshold': threshold,
            'consecutive_speech': self.consecutive_speech_chunks,
            'consecutive_silence': self.consecutive_silence_chunks,
            'baseline_noise': self.echo_canceller.baseline_noise
        }