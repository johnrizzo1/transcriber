"""
Unit tests for Voice Activity Detection (VAD) components.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from transcriber.audio.vad import (
    VoiceActivityDetector,
    SpeechSegmenter,
    VADProcessor
)


@pytest.mark.unit
class TestVoiceActivityDetector:
    """Test VoiceActivityDetector class."""
    
    def test_initialization_default_params(self):
        """Test VAD initialization with default parameters."""
        vad = VoiceActivityDetector()
        
        assert vad.sample_rate == 16000
        assert vad.frame_duration == 30
        assert vad.frame_size == 480  # 16000 * 30 / 1000
        assert vad.speech_threshold == 0.5
        assert vad.silence_duration == 1.0
        assert vad.is_speaking is False
        assert len(vad.speech_frames) == 0
        assert vad.silence_frames == 0
    
    def test_initialization_custom_params(self):
        """Test VAD initialization with custom parameters."""
        vad = VoiceActivityDetector(
            sample_rate=8000,
            frame_duration=20,
            aggressiveness=2,
            speech_threshold=0.7,
            silence_duration=2.0
        )
        
        assert vad.sample_rate == 8000
        assert vad.frame_duration == 20
        assert vad.frame_size == 160  # 8000 * 20 / 1000
        assert vad.speech_threshold == 0.7
        assert vad.silence_duration == 2.0
    
    def test_invalid_sample_rate(self):
        """Test VAD initialization with invalid sample rate."""
        with pytest.raises(ValueError, match="Sample rate .* not supported"):
            VoiceActivityDetector(sample_rate=11025)
    
    def test_invalid_frame_duration(self):
        """Test VAD initialization with invalid frame duration."""
        with pytest.raises(ValueError, match="Frame duration .* not supported"):
            VoiceActivityDetector(frame_duration=15)
    
    @patch('transcriber.audio.vad.webrtcvad.Vad')
    def test_process_audio_speech_detection(self, mock_vad_class):
        """Test audio processing with speech detection."""
        # Setup mock VAD
        mock_vad = MagicMock()
        mock_vad_class.return_value = mock_vad
        mock_vad.is_speech.return_value = True
        
        vad = VoiceActivityDetector()
        
        # Create test audio (1 second at 16kHz)
        audio_data = np.random.randn(16000).astype(np.float32)
        
        # Process audio
        is_complete, speech_segment = vad.process_audio(audio_data)
        
        # Should detect speech but not complete yet
        assert is_complete is False
        assert speech_segment is None
        assert vad.is_speaking is True
        assert len(vad.speech_frames) > 0
    
    @patch('transcriber.audio.vad.webrtcvad.Vad')
    def test_process_audio_silence_detection(self, mock_vad_class):
        """Test audio processing with silence detection."""
        # Setup mock VAD
        mock_vad = MagicMock()
        mock_vad_class.return_value = mock_vad
        mock_vad.is_speech.return_value = False
        
        vad = VoiceActivityDetector(silence_duration=0.1)  # Short silence
        
        # First, simulate speech
        vad.is_speaking = True
        vad.speech_frames.append(np.random.randn(1600).astype(np.float32))
        
        # Create silence audio
        silence_data = np.zeros(1600, dtype=np.float32)
        
        # Process silence - should complete speech segment
        is_complete, speech_segment = vad.process_audio(silence_data)
        
        # Should complete speech segment
        assert is_complete is True
        assert speech_segment is not None
        assert isinstance(speech_segment, np.ndarray)
        assert vad.is_speaking is False
        assert len(vad.speech_frames) == 0
    
    @patch('transcriber.audio.vad.webrtcvad.Vad')
    def test_process_audio_incomplete_frames(self, mock_vad_class):
        """Test audio processing with incomplete frames."""
        mock_vad = MagicMock()
        mock_vad_class.return_value = mock_vad
        
        vad = VoiceActivityDetector()
        
        # Create audio shorter than frame size
        short_audio = np.random.randn(100).astype(np.float32)
        
        # Process short audio
        is_complete, speech_segment = vad.process_audio(short_audio)
        
        # Should not process incomplete frames
        assert is_complete is False
        assert speech_segment is None
        assert not mock_vad.is_speech.called
    
    @patch('transcriber.audio.vad.webrtcvad.Vad')
    def test_vad_error_handling(self, mock_vad_class):
        """Test VAD error handling."""
        mock_vad = MagicMock()
        mock_vad_class.return_value = mock_vad
        mock_vad.is_speech.side_effect = Exception("VAD error")
        
        vad = VoiceActivityDetector()
        audio_data = np.random.randn(16000).astype(np.float32)
        
        # Should handle VAD errors gracefully
        is_complete, speech_segment = vad.process_audio(audio_data)
        
        assert is_complete is False
        assert speech_segment is None
    
    def test_reset_state(self):
        """Test VAD state reset."""
        vad = VoiceActivityDetector()
        
        # Set some state
        vad.is_speaking = True
        vad.speech_frames.append(np.random.randn(1000))
        vad.silence_frames = 5
        vad.frame_decisions.append(True)
        vad.speech_start_time = 1.0
        
        # Reset state
        vad.reset()
        
        # Check state is reset
        assert vad.is_speaking is False
        assert len(vad.speech_frames) == 0
        assert vad.silence_frames == 0
        assert len(vad.frame_decisions) == 0
        assert vad.speech_start_time is None
    
    def test_get_state(self):
        """Test getting VAD state for debugging."""
        vad = VoiceActivityDetector()
        
        # Set some state
        vad.is_speaking = True
        vad.speech_frames.append(np.random.randn(1000))
        vad.silence_frames = 3
        vad.frame_decisions.extend([True, False, True])
        
        state = vad.get_state()
        
        assert state['is_speaking'] is True
        assert state['buffered_frames'] == 1
        assert state['silence_frames'] == 3
        assert state['recent_decisions'] == [True, False, True]


@pytest.mark.unit
class TestSpeechSegmenter:
    """Test SpeechSegmenter class."""
    
    @patch('transcriber.audio.vad.VoiceActivityDetector')
    def test_initialization(self, mock_vad_class):
        """Test speech segmenter initialization."""
        segmenter = SpeechSegmenter()
        
        # Should create VAD with default config
        mock_vad_class.assert_called_once_with()
        assert len(segmenter.segments) == 0
        assert len(segmenter._audio_buffer) == 0
        assert segmenter._start_time == 0
    
    @patch('transcriber.audio.vad.VoiceActivityDetector')
    def test_initialization_with_config(self, mock_vad_class):
        """Test speech segmenter initialization with custom config."""
        vad_config = {'speech_threshold': 0.7, 'silence_duration': 2.0}
        segmenter = SpeechSegmenter(vad_config)
        
        mock_vad_class.assert_called_once_with(**vad_config)
    
    def test_process_chunk_no_speech(self):
        """Test processing audio chunk with no speech detected."""
        segmenter = SpeechSegmenter()
        segmenter.vad.process_audio = MagicMock(return_value=(False, None))
        
        audio_chunk = np.random.randn(1600).astype(np.float32)
        result = segmenter.process_chunk(audio_chunk)
        
        assert result is None
        assert len(segmenter._audio_buffer) == 1
        segmenter.vad.process_audio.assert_called_once_with(audio_chunk)
    
    def test_process_chunk_with_speech(self):
        """Test processing audio chunk with speech detected."""
        segmenter = SpeechSegmenter()
        
        # Mock VAD to return speech segment
        speech_audio = np.random.randn(8000).astype(np.float32)
        segmenter.vad.process_audio = MagicMock(
            return_value=(True, speech_audio)
        )
        segmenter.vad.sample_rate = 16000
        
        audio_chunk = np.random.randn(1600).astype(np.float32)
        result = segmenter.process_chunk(audio_chunk)
        
        assert result is not None
        assert isinstance(result, dict)
        assert 'audio' in result
        assert 'start_time' in result
        assert 'end_time' in result
        assert 'duration' in result
        assert 'sample_rate' in result
        
        assert np.array_equal(result['audio'], speech_audio)
        assert result['sample_rate'] == 16000
        assert result['duration'] == len(speech_audio) / 16000
        assert len(segmenter.segments) == 1
    
    def test_get_segments(self):
        """Test getting all detected segments."""
        segmenter = SpeechSegmenter()
        
        # Add some mock segments
        segment1 = {'audio': np.random.randn(1000), 'duration': 1.0}
        segment2 = {'audio': np.random.randn(2000), 'duration': 2.0}
        segmenter.segments = [segment1, segment2]
        
        segments = segmenter.get_segments()
        
        assert len(segments) == 2
        assert segments == [segment1, segment2]
        # Should return a copy
        assert segments is not segmenter.segments
    
    def test_reset(self):
        """Test segmenter reset."""
        segmenter = SpeechSegmenter()
        segmenter.vad.reset = MagicMock()
        
        # Add some state
        segmenter.segments.append({'audio': np.random.randn(1000)})
        segmenter._audio_buffer.append(np.random.randn(1000))
        segmenter._start_time = 5
        
        segmenter.reset()
        
        # Check state is reset
        assert len(segmenter.segments) == 0
        assert len(segmenter._audio_buffer) == 0
        assert segmenter._start_time == 0
        segmenter.vad.reset.assert_called_once()


@pytest.mark.unit
class TestVADProcessor:
    """Test VADProcessor class."""
    
    def test_initialization(self):
        """Test VAD processor initialization."""
        processor = VADProcessor()
        
        assert processor.vad_threshold == 0.5
        assert isinstance(processor.segmenter, SpeechSegmenter)
    
    def test_initialization_with_threshold(self):
        """Test VAD processor initialization with custom threshold."""
        processor = VADProcessor(vad_threshold=0.8)
        
        assert processor.vad_threshold == 0.8
    
    @pytest.mark.asyncio
    async def test_process_audio_stream(self):
        """Test processing audio stream."""
        processor = VADProcessor()
        
        # Mock segmenter
        processor.segmenter.process_chunk = MagicMock()
        
        # Create mock audio stream
        async def mock_audio_stream():
            for i in range(3):
                yield np.random.randn(1600).astype(np.float32)
        
        # Mock segmenter to return speech on second chunk
        speech_audio = np.random.randn(8000).astype(np.float32)
        processor.segmenter.process_chunk.side_effect = [
            None,  # First chunk - no speech
            {'audio': speech_audio},  # Second chunk - speech detected
            None   # Third chunk - no speech
        ]
        
        # Process stream
        speech_segments = []
        async for segment in processor.process_audio_stream(mock_audio_stream()):
            speech_segments.append(segment)
        
        # Should yield one speech segment
        assert len(speech_segments) == 1
        assert np.array_equal(speech_segments[0], speech_audio)
        assert processor.segmenter.process_chunk.call_count == 3
    
    @pytest.mark.asyncio
    async def test_process_audio_stream_no_speech(self):
        """Test processing audio stream with no speech."""
        processor = VADProcessor()
        processor.segmenter.process_chunk = MagicMock(return_value=None)
        
        # Create mock audio stream
        async def mock_audio_stream():
            for i in range(2):
                yield np.random.randn(1600).astype(np.float32)
        
        # Process stream
        speech_segments = []
        async for segment in processor.process_audio_stream(mock_audio_stream()):
            speech_segments.append(segment)
        
        # Should yield no speech segments
        assert len(speech_segments) == 0
        assert processor.segmenter.process_chunk.call_count == 2
    
    def test_reset(self):
        """Test processor reset."""
        processor = VADProcessor()
        processor.segmenter.reset = MagicMock()
        
        processor.reset()
        
        processor.segmenter.reset.assert_called_once()


@pytest.mark.unit
class TestVADIntegration:
    """Test VAD component integration."""
    
    @patch('transcriber.audio.vad.webrtcvad.Vad')
    def test_complete_vad_pipeline(self, mock_vad_class):
        """Test complete VAD pipeline from audio to speech segments."""
        # Setup mock VAD
        mock_vad = MagicMock()
        mock_vad_class.return_value = mock_vad
        
        # Simulate speech detection pattern: speech -> speech -> silence
        mock_vad.is_speech.side_effect = [True, True, False]
        
        vad = VoiceActivityDetector(silence_duration=0.1)
        
        # Process speech chunks
        audio_chunk1 = np.random.randn(1600).astype(np.float32)
        audio_chunk2 = np.random.randn(1600).astype(np.float32)
        silence_chunk = np.zeros(1600, dtype=np.float32)
        
        # First chunk - start speech
        is_complete1, segment1 = vad.process_audio(audio_chunk1)
        assert is_complete1 is False
        assert segment1 is None
        assert vad.is_speaking is True
        
        # Second chunk - continue speech
        is_complete2, segment2 = vad.process_audio(audio_chunk2)
        assert is_complete2 is False
        assert segment2 is None
        assert vad.is_speaking is True
        
        # Silence chunk - end speech
        is_complete3, segment3 = vad.process_audio(silence_chunk)
        assert is_complete3 is True
        assert segment3 is not None
        assert isinstance(segment3, np.ndarray)
        assert vad.is_speaking is False
        
        # Speech segment should contain all speech chunks plus silence
        expected_length = len(audio_chunk1) + len(audio_chunk2) + len(silence_chunk)
        assert len(segment3) == expected_length
    
    def test_speech_segmenter_timing_calculation(self):
        """Test speech segmenter timing calculations."""
        segmenter = SpeechSegmenter()
        segmenter.vad.sample_rate = 16000
        
        # Mock VAD to return speech after processing some chunks
        speech_audio = np.random.randn(8000).astype(np.float32)  # 0.5 seconds
        segmenter.vad.process_audio = MagicMock(return_value=(True, speech_audio))
        
        # Process multiple chunks to build up timing
        for i in range(3):
            chunk = np.random.randn(1600).astype(np.float32)  # 0.1 seconds each
            segmenter._audio_buffer.append(chunk)
        
        # Process final chunk that triggers speech detection
        final_chunk = np.random.randn(1600).astype(np.float32)
        result = segmenter.process_chunk(final_chunk)
        
        assert result is not None
        
        # Check timing calculations
        expected_end_time = 4 * 1600 / 16000  # 4 chunks * 0.1 seconds
        expected_duration = len(speech_audio) / 16000  # 0.5 seconds
        expected_start_time = expected_end_time - expected_duration
        
        assert abs(result['end_time'] - expected_end_time) < 0.001
        assert abs(result['duration'] - expected_duration) < 0.001
        assert abs(result['start_time'] - expected_start_time) < 0.001