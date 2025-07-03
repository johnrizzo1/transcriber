"""
Stress tests for system limits and performance under load.
"""

import asyncio
import time
import pytest
import psutil
import os

from transcriber.audio.vad import VoiceActivityDetector
from transcriber.tools.registry import ToolRegistry
from transcriber.tools.base import ToolResult
from transcriber.session.manager import SessionManager
from transcriber.session.models import MessageType


@pytest.mark.performance
@pytest.mark.slow
class TestStressTests:
    """Stress tests for system limits."""
    
    def test_vad_continuous_processing(self, mock_audio_data):
        """Test VAD under continuous processing load."""
        vad = VoiceActivityDetector()
        
        # Process audio continuously for extended period
        start_time = time.time()
        chunk_size = 1600  # 100ms at 16kHz
        processed_chunks = 0
        
        # Process for 5 seconds worth of audio data
        target_duration = 5.0
        while (time.time() - start_time) < target_duration:
            for i in range(0, len(mock_audio_data), chunk_size):
                chunk = mock_audio_data[i:i + chunk_size]
                if len(chunk) == chunk_size:
                    vad.process_audio(chunk)
                    processed_chunks += 1
                
                # Check if we've hit our time limit
                if (time.time() - start_time) >= target_duration:
                    break
        
        processing_time = time.time() - start_time
        throughput = processed_chunks / processing_time
        
        # Should maintain reasonable throughput
        assert throughput > 50  # At least 50 chunks per second
        print(f"VAD processed {processed_chunks} chunks in "
              f"{processing_time:.2f}s ({throughput:.1f} chunks/s)")
    
    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(self):
        """Test concurrent tool execution under load."""
        registry = ToolRegistry()
        
        # Register a simple mock tool
        from tests.unit.test_tools_registry import MockCalculatorTool
        tool = MockCalculatorTool()
        registry.register(tool)
        
        async def execute_tool_batch(batch_size: int):
            """Execute a batch of tools concurrently."""
            tasks = []
            for i in range(batch_size):
                task = registry.execute_tool("calculator",
                                             expression=f"{i}+{i}")
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successful executions
            successful = 0
            for r in results:
                if isinstance(r, ToolResult) and r.success:
                    successful += 1
            return successful, len(results)
        
        # Test increasing batch sizes
        batch_sizes = [10, 25, 50, 100]
        
        for batch_size in batch_sizes:
            start_time = time.time()
            successful, total = await execute_tool_batch(batch_size)
            execution_time = time.time() - start_time
            
            # All executions should succeed
            assert successful == total == batch_size
            
            # Should complete within reasonable time
            assert execution_time < 10.0  # < 10 seconds
            
            throughput = batch_size / execution_time
            print(f"Batch size {batch_size}: {throughput:.1f} tools/sec")
    
    @pytest.mark.asyncio
    async def test_session_high_volume(self, temp_dir):
        """Test session management with high message volume."""
        manager = SessionManager(data_dir=str(temp_dir))
        await manager.initialize()
        
        try:
            # Create session
            session = await manager.start_new_session("High Volume Test")
            
            # Add messages in batches
            batch_size = 100
            total_messages = 1000
            
            start_time = time.time()
            
            for batch in range(0, total_messages, batch_size):
                batch_start = time.time()
                
                # Add batch of messages
                for i in range(batch, min(batch + batch_size, total_messages)):
                    await manager.add_message_to_current_session(
                        f"High volume message {i}",
                        MessageType.USER if i % 2 == 0 else MessageType.ASSISTANT
                    )
                
                batch_time = time.time() - batch_start
                print(f"Batch {batch//batch_size + 1}: "
                      f"{batch_size/batch_time:.1f} msgs/sec")
            
            total_time = time.time() - start_time
            
            # Verify all messages were added
            final_session = await manager.get_session(session.id)
            assert final_session is not None
            assert len(final_session.messages) == total_messages
            
            # Performance should be reasonable
            throughput = total_messages / total_time
            assert throughput > 50  # At least 50 messages per second
            
            print(f"Total: {total_messages} messages in {total_time:.2f}s "
                  f"({throughput:.1f} msgs/sec)")
            
        finally:
            await manager.close()
    
    def test_memory_stability_under_load(self, mock_audio_data):
        """Test memory stability under sustained load."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        vad = VoiceActivityDetector()
        registry = ToolRegistry()
        
        # Register tool for testing
        from tests.unit.test_tools_registry import MockCalculatorTool
        tool = MockCalculatorTool()
        registry.register(tool)
        
        # Run sustained operations
        iterations = 50
        chunk_size = 1600
        
        for iteration in range(iterations):
            # Process audio
            for i in range(0, len(mock_audio_data), chunk_size):
                chunk = mock_audio_data[i:i + chunk_size]
                if len(chunk) == chunk_size:
                    vad.process_audio(chunk)
            
            # Execute tools
            asyncio.run(self._execute_tool_batch(registry, 10))
            
            # Check memory every 10 iterations
            if iteration % 10 == 0:
                current_memory = process.memory_info().rss
                memory_growth = current_memory - initial_memory
                memory_growth_mb = memory_growth / (1024 * 1024)
                
                print(f"Iteration {iteration}: "
                      f"Memory growth: {memory_growth_mb:.1f}MB")
                
                # Memory growth should be bounded
                max_growth_mb = 100  # 100MB maximum growth
                if memory_growth_mb > max_growth_mb:
                    pytest.fail(f"Excessive memory growth: "
                              f"{memory_growth_mb:.1f}MB")
        
        final_memory = process.memory_info().rss
        total_growth = (final_memory - initial_memory) / (1024 * 1024)
        
        print(f"Final memory growth: {total_growth:.1f}MB")
        
        # Total growth should be reasonable
        assert total_growth < 100  # Less than 100MB total growth
    
    async def _execute_tool_batch(self, registry: ToolRegistry, count: int):
        """Helper to execute a batch of tools."""
        tasks = []
        for i in range(count):
            task = registry.execute_tool("calculator", expression="1+1")
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    @pytest.mark.asyncio
    async def test_rapid_session_creation(self, temp_dir):
        """Test rapid session creation and cleanup."""
        manager = SessionManager(data_dir=str(temp_dir))
        await manager.initialize()
        
        try:
            session_count = 50
            start_time = time.time()
            
            created_sessions = []
            
            # Create sessions rapidly
            for i in range(session_count):
                session = await manager.start_new_session(f"Rapid Test {i}")
                created_sessions.append(session.id)
                
                # Add a few messages to each
                for j in range(5):
                    await manager.add_message_to_current_session(
                        f"Message {j}",
                        MessageType.USER if j % 2 == 0 else MessageType.ASSISTANT
                    )
            
            creation_time = time.time() - start_time
            
            # Verify all sessions exist
            for session_id in created_sessions:
                session = await manager.get_session(session_id)
                assert session is not None
                assert len(session.messages) == 5
            
            # Performance should be reasonable
            throughput = session_count / creation_time
            assert throughput > 5  # At least 5 sessions per second
            
            print(f"Created {session_count} sessions in "
                  f"{creation_time:.2f}s ({throughput:.1f} sessions/sec)")
            
        finally:
            await manager.close()


@pytest.mark.performance
class TestPerformanceRegression:
    """Regression tests to track performance over time."""
    
    def test_vad_baseline_performance(self, mock_audio_data):
        """Baseline VAD performance test."""
        vad = VoiceActivityDetector()
        
        # Process standard amount of audio
        iterations = 10
        chunk_size = 1600
        
        start_time = time.time()
        
        for _ in range(iterations):
            for i in range(0, len(mock_audio_data), chunk_size):
                chunk = mock_audio_data[i:i + chunk_size]
                if len(chunk) == chunk_size:
                    vad.process_audio(chunk)
        
        total_time = time.time() - start_time
        
        # Calculate throughput
        total_samples = iterations * len(mock_audio_data)
        audio_duration = total_samples / 16000  # 16kHz sample rate
        real_time_factor = audio_duration / total_time
        
        # Should process faster than real-time
        assert real_time_factor > 1.0
        
        print(f"VAD Baseline: {real_time_factor:.1f}x real-time processing")
        
        # Log for regression tracking
        with open("performance_baseline.txt", "a") as f:
            f.write(f"VAD_BASELINE: {real_time_factor:.2f}x\n")
    
    @pytest.mark.asyncio
    async def test_tool_execution_baseline(self):
        """Baseline tool execution performance."""
        registry = ToolRegistry()
        
        from tests.unit.test_tools_registry import MockCalculatorTool
        tool = MockCalculatorTool()
        registry.register(tool)
        
        # Execute tools and measure performance
        iterations = 100
        start_time = time.time()
        
        for _ in range(iterations):
            result = await registry.execute_tool("calculator", 
                                               expression="2+2")
            assert result.success is True
        
        total_time = time.time() - start_time
        throughput = iterations / total_time
        
        # Should execute many tools per second
        assert throughput > 50
        
        print(f"Tool Execution Baseline: {throughput:.1f} executions/sec")
        
        # Log for regression tracking
        with open("performance_baseline.txt", "a") as f:
            f.write(f"TOOL_BASELINE: {throughput:.1f} exec/sec\n")