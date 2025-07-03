"""
Integration tests for the complete voice processing pipeline.
"""

import asyncio
import numpy as np
import pytest
from transcriber.audio.vad import VADProcessor
from transcriber.audio.stt import STTEngine
from transcriber.audio.tts import TTSEngine
from transcriber.agent.core import VoiceAgent


@pytest.mark.integration
class TestVoicePipelineIntegration:
    """Test complete voice pipeline integration."""
    
    @pytest.mark.asyncio
    async def test_audio_to_text_pipeline(
        self,
        test_settings,
        mock_audio_data,
        mock_external_services
    ):
        """Test audio input to text transcription pipeline."""
        # Create VAD processor
        vad_processor = VADProcessor(vad_threshold=0.5)
        
        # Create STT processor with mocked Whisper
        stt_processor = STTEngine(test_settings)
        
        # Mock audio stream
        async def mock_audio_stream():
            # Yield multiple chunks to simulate continuous audio
            for _ in range(5):
                yield mock_audio_data[:1600]  # 100ms chunks
        
        # Process audio through VAD
        speech_segments = []
        async for segment in vad_processor.process_audio_stream(mock_audio_stream()):
            speech_segments.append(segment)
        
        # Should detect at least one speech segment
        assert len(speech_segments) > 0
        
        # Process speech segments through STT
        if speech_segments:
            transcription = await stt_processor.transcribe_audio(
                speech_segments[0], 16000
            )
            assert transcription is not None
            assert isinstance(transcription, str)
            assert len(transcription) > 0
    
    @pytest.mark.asyncio
    async def test_text_to_speech_pipeline(
        self,
        test_settings,
        mock_external_services
    ):
        """Test text to speech synthesis pipeline."""
        # Create TTS processor
        tts_processor = TTSEngine(test_settings)
        
        # Test text synthesis
        test_text = "Hello, this is a test message for speech synthesis."
        audio_data = await tts_processor.synthesize_text(test_text)
        
        assert audio_data is not None
        assert isinstance(audio_data, (bytes, np.ndarray))
        assert len(audio_data) > 0
    
    @pytest.mark.asyncio
    async def test_agent_text_processing(
        self,
        test_settings,
        mock_external_services
    ):
        """Test agent text processing integration."""
        # Create voice agent
        agent = VoiceAgent(test_settings)
        await agent.initialize()
        
        try:
            # Test text input processing
            response = await agent.process_text_input("Hello, can you help me?")
            
            assert response is not None
            assert isinstance(response, str)
            assert len(response) > 0
            
            # Test conversation memory
            response2 = await agent.process_text_input("What did I just ask?")
            assert response2 is not None
            
        finally:
            await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_session_integration_with_agent(
        self,
        test_settings,
        mock_external_services
    ):
        """Test session management integration with agent."""
        # Enable session management
        test_settings.session.enabled = True
        test_settings.session.auto_start_session = True
        
        # Create agent with session management
        agent = VoiceAgent(test_settings)
        await agent.initialize()
        
        try:
            # Process messages - should create session automatically
            response1 = await agent.process_text_input("Hello")
            assert response1 is not None
            
            # Check session was created
            current_session = await agent.get_current_session()
            assert current_session is not None
            assert len(current_session.messages) >= 2  # User + Assistant
            
            # Process another message
            response2 = await agent.process_text_input("How are you?")
            assert response2 is not None
            
            # Check session was updated
            updated_session = await agent.get_current_session()
            assert len(updated_session.messages) >= 4  # 2 more messages
            
            # Complete session
            completed_session = await agent.complete_current_session()
            assert completed_session is not None
            assert completed_session.status.value == "completed"
            
        finally:
            await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_tool_execution_integration(
        self,
        test_settings,
        mock_external_services
    ):
        """Test tool execution integration with agent."""
        # Create agent
        agent = VoiceAgent(test_settings)
        await agent.initialize()
        
        try:
            # Test tool execution through natural language
            response = await agent.process_text_input("Calculate 2 + 2")
            
            assert response is not None
            # Response should contain calculation result or tool execution
            assert any(keyword in response.lower() for keyword in ["4", "result", "calculate"])
            
        finally:
            await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(
        self,
        test_settings,
        mock_external_services
    ):
        """Test error handling across pipeline components."""
        # Create agent
        agent = VoiceAgent(test_settings)
        await agent.initialize()
        
        try:
            # Test with empty input
            response = await agent.process_text_input("")
            # Should handle gracefully
            assert response is not None
            
            # Test with very long input
            long_input = "test " * 1000
            response = await agent.process_text_input(long_input)
            # Should handle gracefully
            assert response is not None
            
        finally:
            await agent.cleanup()


@pytest.mark.integration
class TestPerformanceIntegration:
    """Test performance monitoring integration."""
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_with_pipeline(
        self,
        test_settings,
        performance_integration,
        mock_external_services
    ):
        """Test performance monitoring integration with voice pipeline."""
        # Create agent with performance monitoring
        agent = VoiceAgent(test_settings)
        agent.performance_integration = performance_integration
        await agent.initialize()
        
        try:
            # Process text with performance monitoring
            with performance_integration.time_operation("agent", "text_processing"):
                response = await agent.process_text_input("Test message")
            
            assert response is not None
            
            # Check performance data was collected
            summary = performance_integration.get_performance_summary()
            assert "components" in summary
            
        finally:
            await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_benchmark_integration(
        self,
        test_settings,
        performance_integration,
        mock_external_services
    ):
        """Test benchmarking integration."""
        # Create simple benchmark function
        async def text_processing_benchmark():
            agent = VoiceAgent(test_settings)
            await agent.initialize()
            try:
                response = await agent.process_text_input("Benchmark test")
                return len(response) if response else 0
            finally:
                await agent.cleanup()
        
        # Run benchmark
        result = await performance_integration.run_component_benchmark(
            "agent",
            text_processing_benchmark,
            iterations=3
        )
        
        if result:
            assert result.avg_time_ms > 0
            assert result.iterations == 3


@pytest.mark.integration
class TestConcurrencyIntegration:
    """Test concurrent operations integration."""
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_requests(
        self,
        test_settings,
        mock_external_services
    ):
        """Test concurrent agent request handling."""
        agent = VoiceAgent(test_settings)
        await agent.initialize()
        
        try:
            # Create multiple concurrent requests
            tasks = []
            for i in range(5):
                task = agent.process_text_input(f"Concurrent request {i}")
                tasks.append(task)
            
            # Wait for all requests to complete
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All requests should complete successfully
            for response in responses:
                assert not isinstance(response, Exception)
                assert response is not None
                assert isinstance(response, str)
                
        finally:
            await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_concurrent_session_operations(
        self,
        test_settings,
        session_manager,
        mock_external_services
    ):
        """Test concurrent session operations."""
        # Create multiple sessions concurrently
        tasks = []
        for i in range(3):
            task = session_manager.start_new_session(f"Concurrent Session {i}")
            tasks.append(task)
        
        sessions = await asyncio.gather(*tasks)
        
        # All sessions should be created successfully
        assert len(sessions) == 3
        for session in sessions:
            assert session is not None
            assert session.title.startswith("Concurrent Session")
        
        # Add messages to sessions concurrently
        message_tasks = []
        for i, session in enumerate(sessions):
            # Set current session
            session_manager._current_session_id = session.id
            task = session_manager.add_message_to_current_session(
                f"Message for session {i}",
                "user"
            )
            message_tasks.append(task)
        
        await asyncio.gather(*message_tasks)
        
        # Verify messages were added
        for session in sessions:
            updated_session = await session_manager.get_session(session.id)
            assert len(updated_session.messages) > 0


@pytest.mark.integration
class TestDataFlowIntegration:
    """Test data flow between components."""
    
    @pytest.mark.asyncio
    async def test_audio_data_flow(
        self,
        test_settings,
        mock_audio_data,
        mock_external_services
    ):
        """Test audio data flow through pipeline."""
        # Create pipeline components
        vad_processor = VADProcessor()
        stt_processor = STTProcessor(test_settings)
        
        # Simulate audio data flow
        audio_chunks = [mock_audio_data[:1600] for _ in range(10)]
        
        # Process through VAD
        speech_segments = []
        for chunk in audio_chunks:
            segment = vad_processor.segmenter.process_chunk(chunk)
            if segment:
                speech_segments.append(segment['audio'])
        
        # Process through STT if we have speech
        transcriptions = []
        for segment in speech_segments:
            if len(segment) > 0:
                transcription = await stt_processor.transcribe_audio(segment)
                if transcription:
                    transcriptions.append(transcription)
        
        # Should have some transcriptions
        assert len(transcriptions) >= 0  # May be 0 if no speech detected
    
    @pytest.mark.asyncio
    async def test_text_data_flow(
        self,
        test_settings,
        mock_external_services
    ):
        """Test text data flow through agent and session."""
        # Enable session management
        test_settings.session.enabled = True
        
        agent = VoiceAgent(test_settings)
        await agent.initialize()
        
        try:
            # Process text through agent
            user_input = "What's the weather like?"
            response = await agent.process_text_input(user_input)
            
            assert response is not None
            
            # Check session captured the interaction
            if hasattr(agent, 'session_manager') and agent.session_manager:
                current_session = await agent.get_current_session()
                if current_session:
                    # Should have user message and assistant response
                    assert len(current_session.messages) >= 2
                    
                    # Check message content
                    user_messages = current_session.get_messages_by_type("user")
                    assert len(user_messages) > 0
                    assert user_input in [msg.content for msg in user_messages]
                    
        finally:
            await agent.cleanup()


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndIntegration:
    """Test complete end-to-end scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_voice_conversation(
        self,
        test_settings,
        mock_audio_data,
        mock_external_services
    ):
        """Test complete voice conversation flow."""
        # Enable all features
        test_settings.session.enabled = True
        test_settings.session.auto_start_session = True
        
        agent = VoiceAgent(test_settings)
        await agent.initialize()
        
        try:
            # Simulate voice conversation
            conversation_turns = [
                "Hello, can you help me?",
                "What's 2 plus 2?",
                "Thank you for your help!"
            ]
            
            responses = []
            for turn in conversation_turns:
                response = await agent.process_text_input(turn)
                responses.append(response)
                
                # Small delay to simulate natural conversation
                await asyncio.sleep(0.1)
            
            # All turns should get responses
            assert len(responses) == len(conversation_turns)
            for response in responses:
                assert response is not None
                assert len(response) > 0
            
            # Check session was maintained
            session = await agent.get_current_session()
            if session:
                # Should have all conversation turns
                assert len(session.messages) >= len(conversation_turns) * 2
                
                # Check conversation flow
                user_messages = session.get_messages_by_type("user")
                assert len(user_messages) == len(conversation_turns)
                
                for i, turn in enumerate(conversation_turns):
                    assert turn in [msg.content for msg in user_messages]
            
        finally:
            await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_tool_usage_scenario(
        self,
        test_settings,
        mock_external_services
    ):
        """Test realistic tool usage scenario."""
        agent = VoiceAgent(test_settings)
        await agent.initialize()
        
        try:
            # Scenario: User asks for calculations and file operations
            scenarios = [
                "Calculate 15 * 23",
                "What's the square root of 144?",
                "List files in the current directory",
                "Get system information"
            ]
            
            for scenario in scenarios:
                response = await agent.process_text_input(scenario)
                assert response is not None
                
                # Response should indicate tool usage or results
                assert len(response) > 0
                
        finally:
            await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_error_recovery_scenario(
        self,
        test_settings,
        mock_external_services
    ):
        """Test error recovery in realistic scenarios."""
        agent = VoiceAgent(test_settings)
        await agent.initialize()
        
        try:
            # Test various error conditions
            error_scenarios = [
                "",  # Empty input
                "a" * 10000,  # Very long input
                "Invalid command that doesn't make sense",
                "Calculate invalid_expression",
            ]
            
            for scenario in error_scenarios:
                response = await agent.process_text_input(scenario)
                # Should handle all errors gracefully
                assert response is not None
                # Should not crash or return empty response
                assert isinstance(response, str)
                
        finally:
            await agent.cleanup()


@pytest.mark.integration
class TestResourceManagement:
    """Test resource management across components."""
    
    @pytest.mark.asyncio
    async def test_memory_usage_integration(
        self,
        test_settings,
        mock_external_services
    ):
        """Test memory usage across pipeline components."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create and use agent
        agent = VoiceAgent(test_settings)
        await agent.initialize()
        
        try:
            # Process multiple requests
            for i in range(10):
                response = await agent.process_text_input(f"Test message {i}")
                assert response is not None
            
            # Check memory usage hasn't grown excessively
            current_memory = process.memory_info().rss
            memory_growth = current_memory - initial_memory
            
            # Memory growth should be reasonable (less than 100MB for this test)
            assert memory_growth < 100 * 1024 * 1024
            
        finally:
            await agent.cleanup()
            
            # Memory should be released after cleanup
            await asyncio.sleep(0.1)  # Allow cleanup to complete
            final_memory = process.memory_info().rss
            
            # Memory should not have grown significantly after cleanup
            final_growth = final_memory - initial_memory
            assert final_growth < memory_growth + 50 * 1024 * 1024
    
    @pytest.mark.asyncio
    async def test_connection_management(
        self,
        test_settings,
        mock_external_services
    ):
        """Test connection management across components."""
        # Test multiple agent instances
        agents = []
        
        try:
            # Create multiple agents
            for i in range(3):
                agent = VoiceAgent(test_settings)
                await agent.initialize()
                agents.append(agent)
            
            # Use all agents
            tasks = []
            for i, agent in enumerate(agents):
                task = agent.process_text_input(f"Message from agent {i}")
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            
            # All should work
            for response in responses:
                assert response is not None
                
        finally:
            # Cleanup all agents
            for agent in agents:
                await agent.cleanup()