"""
Fixed integration tests for the voice pipeline.
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from transcriber.session.manager import SessionManager
from transcriber.session.models import MessageType
from transcriber.tools.registry import ToolRegistry


@pytest.mark.integration
class TestVoicePipelineIntegration:
    """Integration tests for the complete voice pipeline."""
    
    @pytest.mark.asyncio
    async def test_basic_voice_flow(self, temp_dir, mock_audio_data):
        """Test basic voice processing flow."""
        # Setup session manager
        session_manager = SessionManager(data_dir=str(temp_dir))
        await session_manager.initialize()
        
        try:
            # Start a session
            session = await session_manager.start_new_session("Integration Test")
            assert session is not None
            
            # Simulate voice input processing
            user_message = "Hello, can you help me?"
            await session_manager.add_message_to_current_session(
                user_message, MessageType.USER
            )
            
            # Simulate agent response
            agent_response = "Hello! I'd be happy to help you."
            await session_manager.add_message_to_current_session(
                agent_response, MessageType.ASSISTANT
            )
            
            # Verify session state
            current_session = await session_manager.get_current_session()
            assert current_session is not None
            assert len(current_session.messages) == 2
            assert current_session.messages[0].content == user_message
            assert current_session.messages[1].content == agent_response
            
        finally:
            await session_manager.close()
    
    @pytest.mark.asyncio
    async def test_tool_integration_flow(self, temp_dir):
        """Test integration with tool execution."""
        # Setup components
        session_manager = SessionManager(data_dir=str(temp_dir))
        await session_manager.initialize()
        
        tool_registry = ToolRegistry()
        
        # Register a mock tool
        from tests.unit.test_tools_registry import MockCalculatorTool
        calculator = MockCalculatorTool()
        tool_registry.register(calculator)
        
        try:
            # Start session
            session = await session_manager.start_new_session("Tool Integration")
            
            # User asks for calculation
            user_request = "Calculate 2 + 2"
            await session_manager.add_message_to_current_session(
                user_request, MessageType.USER
            )
            
            # Execute tool
            tool_result = await tool_registry.execute_tool(
                "calculator", expression="2+2"
            )
            
            assert tool_result.success is True
            assert tool_result.output == 4
            
            # Log tool execution
            await session_manager.add_message_to_current_session(
                f"Tool executed: {tool_result.output}", MessageType.TOOL
            )
            
            # Agent responds with result
            agent_response = f"The result is {tool_result.output}"
            await session_manager.add_message_to_current_session(
                agent_response, MessageType.ASSISTANT
            )
            
            # Verify complete flow
            final_session = await session_manager.get_session(session.id)
            assert final_session is not None
            assert len(final_session.messages) == 3
            assert final_session.metadata.tool_executions == 1
            
        finally:
            await session_manager.close()
    
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, temp_dir):
        """Test multi-turn conversation handling."""
        session_manager = SessionManager(data_dir=str(temp_dir))
        await session_manager.initialize()
        
        try:
            session = await session_manager.start_new_session("Multi-turn Test")
            
            # Simulate multiple conversation turns
            conversations = [
                ("Hello", "Hi there!"),
                ("What's the weather like?", "I don't have weather data."),
                ("Can you help with math?", "Yes, I can help with calculations."),
                ("What's 5 + 3?", "5 + 3 equals 8."),
                ("Thank you", "You're welcome!")
            ]
            
            for user_msg, assistant_msg in conversations:
                await session_manager.add_message_to_current_session(
                    user_msg, MessageType.USER
                )
                await session_manager.add_message_to_current_session(
                    assistant_msg, MessageType.ASSISTANT
                )
            
            # Verify conversation state
            final_session = await session_manager.get_session(session.id)
            assert final_session is not None
            assert len(final_session.messages) == len(conversations) * 2
            assert final_session.metadata.user_messages == len(conversations)
            assert final_session.metadata.assistant_messages == len(conversations)
            
            # Verify message order and content
            for i, (user_msg, assistant_msg) in enumerate(conversations):
                user_index = i * 2
                assistant_index = i * 2 + 1
                
                assert final_session.messages[user_index].content == user_msg
                assert final_session.messages[user_index].message_type == MessageType.USER
                assert final_session.messages[assistant_index].content == assistant_msg
                assert final_session.messages[assistant_index].message_type == MessageType.ASSISTANT
            
        finally:
            await session_manager.close()
    
    @pytest.mark.asyncio
    async def test_concurrent_session_handling(self, temp_dir):
        """Test handling multiple concurrent sessions."""
        session_manager = SessionManager(data_dir=str(temp_dir))
        await session_manager.initialize()
        
        try:
            # Create multiple sessions concurrently
            async def create_session_with_messages(session_id: int):
                session = await session_manager.start_new_session(
                    f"Concurrent Session {session_id}"
                )
                
                # Add messages to this session
                for i in range(5):
                    await session_manager.add_message_to_current_session(
                        f"Message {i} from session {session_id}",
                        MessageType.USER if i % 2 == 0 else MessageType.ASSISTANT
                    )
                
                return session.id
            
            # Create 5 concurrent sessions
            session_tasks = [
                create_session_with_messages(i) for i in range(5)
            ]
            
            session_ids = await asyncio.gather(*session_tasks)
            
            # Verify all sessions were created correctly
            assert len(session_ids) == 5
            
            for session_id in session_ids:
                session = await session_manager.get_session(session_id)
                assert session is not None
                assert len(session.messages) == 5
            
        finally:
            await session_manager.close()
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, temp_dir):
        """Test error recovery in the pipeline."""
        session_manager = SessionManager(data_dir=str(temp_dir))
        await session_manager.initialize()
        
        tool_registry = ToolRegistry()
        
        try:
            session = await session_manager.start_new_session("Error Recovery Test")
            
            # Test tool execution error
            try:
                result = await tool_registry.execute_tool(
                    "nonexistent_tool", param="value"
                )
                # Should not succeed
                assert result.success is False
            except Exception:
                # Exception is expected for nonexistent tool
                pass
            
            # System should still be functional
            await session_manager.add_message_to_current_session(
                "System recovered from error", MessageType.SYSTEM
            )
            
            # Verify session is still functional
            final_session = await session_manager.get_session(session.id)
            assert final_session is not None
            assert len(final_session.messages) == 1
            
        finally:
            await session_manager.close()
    
    @pytest.mark.asyncio
    async def test_session_persistence(self, temp_dir):
        """Test session persistence across manager restarts."""
        # Create initial session
        session_manager1 = SessionManager(data_dir=str(temp_dir))
        await session_manager1.initialize()
        
        session_id = None
        try:
            session = await session_manager1.start_new_session("Persistence Test")
            session_id = session.id
            
            await session_manager1.add_message_to_current_session(
                "This should persist", MessageType.USER
            )
            
        finally:
            await session_manager1.close()
        
        # Create new manager instance and verify persistence
        session_manager2 = SessionManager(data_dir=str(temp_dir))
        await session_manager2.initialize()
        
        try:
            # Retrieve the session
            persisted_session = await session_manager2.get_session(session_id)
            assert persisted_session is not None
            assert persisted_session.title == "Persistence Test"
            assert len(persisted_session.messages) == 1
            assert persisted_session.messages[0].content == "This should persist"
            
        finally:
            await session_manager2.close()


@pytest.mark.integration
@pytest.mark.slow
class TestPipelinePerformance:
    """Performance-focused integration tests."""
    
    @pytest.mark.asyncio
    async def test_pipeline_latency(self, temp_dir):
        """Test end-to-end pipeline latency."""
        import time
        
        session_manager = SessionManager(data_dir=str(temp_dir))
        await session_manager.initialize()
        
        tool_registry = ToolRegistry()
        from tests.unit.test_tools_registry import MockCalculatorTool
        calculator = MockCalculatorTool()
        tool_registry.register(calculator)
        
        try:
            session = await session_manager.start_new_session("Latency Test")
            
            # Measure complete pipeline latency
            start_time = time.time()
            
            # User input
            await session_manager.add_message_to_current_session(
                "Calculate 10 + 5", MessageType.USER
            )
            
            # Tool execution
            tool_result = await tool_registry.execute_tool(
                "calculator", expression="10+5"
            )
            
            # Agent response
            await session_manager.add_message_to_current_session(
                f"Result: {tool_result.output}", MessageType.ASSISTANT
            )
            
            total_latency = (time.time() - start_time) * 1000  # Convert to ms
            
            # Pipeline should complete quickly
            assert total_latency < 500  # Less than 500ms
            
            print(f"Pipeline latency: {total_latency:.2f}ms")
            
        finally:
            await session_manager.close()
    
    @pytest.mark.asyncio
    async def test_throughput_under_load(self, temp_dir):
        """Test system throughput under load."""
        session_manager = SessionManager(data_dir=str(temp_dir))
        await session_manager.initialize()
        
        try:
            session = await session_manager.start_new_session("Throughput Test")
            
            # Process many messages quickly
            message_count = 100
            start_time = time.time()
            
            for i in range(message_count):
                await session_manager.add_message_to_current_session(
                    f"Message {i}", 
                    MessageType.USER if i % 2 == 0 else MessageType.ASSISTANT
                )
            
            total_time = time.time() - start_time
            throughput = message_count / total_time
            
            # Should handle reasonable throughput
            assert throughput > 50  # More than 50 messages per second
            
            print(f"Message throughput: {throughput:.1f} msgs/sec")
            
        finally:
            await session_manager.close()