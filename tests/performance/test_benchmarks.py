"""
Performance benchmarks and regression tests.
"""

import asyncio
import time

import pytest

from transcriber.audio.vad import VoiceActivityDetector
from transcriber.tools.registry import ToolRegistry
from transcriber.session.manager import SessionManager
from transcriber.session.models import MessageType
from transcriber.performance.monitor import (
    PerformanceMonitor, ComponentType, PerformanceTimer
)


@pytest.mark.performance
class TestAudioPerformance:
    """Test audio processing performance."""
    
    def test_vad_processing_speed(self, mock_audio_data, performance_thresholds):
        """Test VAD processing speed meets requirements."""
        vad = VoiceActivityDetector()
        
        # Measure processing time for 1 second of audio
        start_time = time.time()
        
        # Process audio in chunks
        chunk_size = 1600  # 100ms at 16kHz
        for i in range(0, len(mock_audio_data), chunk_size):
            chunk = mock_audio_data[i:i + chunk_size]
            if len(chunk) == chunk_size:
                vad.process_audio(chunk)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Should process 1 second of audio in less than threshold
        assert processing_time < performance_thresholds['vad_processing']
        
        # Log performance for monitoring
        print(f"VAD processing time: {processing_time:.2f}ms")
    
    def test_vad_memory_usage(self, mock_audio_data):
        """Test VAD memory usage remains stable."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        vad = VoiceActivityDetector()
        
        # Process multiple chunks
        chunk_size = 1600
        for _ in range(100):  # Process 10 seconds worth
            for i in range(0, len(mock_audio_data), chunk_size):
                chunk = mock_audio_data[i:i + chunk_size]
                if len(chunk) == chunk_size:
                    vad.process_audio(chunk)
        
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be minimal (less than 10MB)
        assert memory_growth < 10 * 1024 * 1024
        print(f"VAD memory growth: {memory_growth / 1024 / 1024:.2f}MB")


@pytest.mark.performance
class TestToolPerformance:
    """Test tool system performance."""
    
    @pytest.mark.asyncio
    async def test_tool_registry_lookup_speed(self):
        """Test tool registry lookup performance."""
        registry = ToolRegistry()
        
        # Register multiple mock tools
        from tests.unit.test_tools_registry import MockCalculatorTool, MockFileOpsTool
        
        tools = []
        for i in range(100):
            tool = MockCalculatorTool()
            tool._metadata.name = f"tool_{i}"
            registry.register(tool)
            tools.append(tool)
        
        # Measure lookup time
        start_time = time.time()
        
        for i in range(1000):  # 1000 lookups
            tool_name = f"tool_{i % 100}"
            retrieved_tool = registry.get(tool_name)
            assert retrieved_tool is not None
        
        lookup_time = (time.time() - start_time) * 1000  # Convert to ms
        avg_lookup_time = lookup_time / 1000
        
        # Average lookup should be very fast (< 0.1ms)
        assert avg_lookup_time < 0.1
        print(f"Average tool lookup time: {avg_lookup_time:.4f}ms")
    
    @pytest.mark.asyncio
    async def test_tool_execution_performance(self):
        """Test tool execution performance."""
        registry = ToolRegistry()
        
        from tests.unit.test_tools_registry import MockCalculatorTool
        tool = MockCalculatorTool()
        registry.register(tool)
        
        # Measure execution time
        execution_times = []
        
        for _ in range(50):
            start_time = time.time()
            result = await registry.execute_tool("calculator", expression="2+2")
            execution_time = (time.time() - start_time) * 1000
            execution_times.append(execution_time)
            
            assert result.success is True
        
        avg_execution_time = sum(execution_times) / len(execution_times)
        max_execution_time = max(execution_times)
        
        # Tool execution should be fast
        assert avg_execution_time < 10  # < 10ms average
        assert max_execution_time < 50  # < 50ms maximum
        
        print(f"Tool execution - Avg: {avg_execution_time:.2f}ms, "
              f"Max: {max_execution_time:.2f}ms")


@pytest.mark.performance
class TestSessionPerformance:
    """Test session management performance."""
    
    @pytest.mark.asyncio
    async def test_session_creation_speed(self, temp_dir):
        """Test session creation performance."""
        manager = SessionManager(data_dir=str(temp_dir))
        await manager.initialize()
        
        try:
            # Measure session creation time
            creation_times = []
            
            for i in range(20):
                start_time = time.time()
                session = await manager.start_new_session(f"Performance Test {i}")
                creation_time = (time.time() - start_time) * 1000
                creation_times.append(creation_time)
                
                assert session is not None
            
            avg_creation_time = sum(creation_times) / len(creation_times)
            max_creation_time = max(creation_times)
            
            # Session creation should be fast
            assert avg_creation_time < 100  # < 100ms average
            assert max_creation_time < 500  # < 500ms maximum
            
            print(f"Session creation - Avg: {avg_creation_time:.2f}ms, "
                  f"Max: {max_creation_time:.2f}ms")
            
        finally:
            await manager.close()
    
    @pytest.mark.asyncio
    async def test_message_insertion_speed(self, temp_dir):
        """Test message insertion performance."""
        manager = SessionManager(data_dir=str(temp_dir))
        await manager.initialize()
        
        try:
            # Create a session
            session = await manager.start_new_session("Message Performance Test")
            
            # Measure message insertion time
            insertion_times = []
            
            for i in range(100):
                start_time = time.time()
                await manager.add_message_to_current_session(
                    f"Performance test message {i}",
                    MessageType.USER
                )
                insertion_time = (time.time() - start_time) * 1000
                insertion_times.append(insertion_time)
            
            avg_insertion_time = sum(insertion_times) / len(insertion_times)
            max_insertion_time = max(insertion_times)
            
            # Message insertion should be fast
            assert avg_insertion_time < 50  # < 50ms average
            assert max_insertion_time < 200  # < 200ms maximum
            
            print(f"Message insertion - Avg: {avg_insertion_time:.2f}ms, "
                  f"Max: {max_insertion_time:.2f}ms")
            
        finally:
            await manager.close()
    
    @pytest.mark.asyncio
    async def test_session_query_performance(self, temp_dir):
        """Test session query performance."""
        manager = SessionManager(data_dir=str(temp_dir))
        await manager.initialize()
        
        try:
            # Create multiple sessions with messages
            session_ids = []
            for i in range(10):
                session = await manager.start_new_session(f"Query Test {i}")
                session_ids.append(session.id)
                
                # Add messages to each session
                for j in range(20):
                    await manager.add_message_to_current_session(
                        f"Message {j} in session {i}",
                        MessageType.USER if j % 2 == 0 else MessageType.ASSISTANT
                    )
            
            # Measure query performance
            query_times = []
            
            for session_id in session_ids:
                start_time = time.time()
                session = await manager.get_session(session_id)
                query_time = (time.time() - start_time) * 1000
                query_times.append(query_time)
                
                assert session is not None
                assert len(session.messages) == 20
            
            avg_query_time = sum(query_times) / len(query_times)
            max_query_time = max(query_times)
            
            # Session queries should be fast
            assert avg_query_time < 100  # < 100ms average
            assert max_query_time < 300  # < 300ms maximum
            
            print(f"Session query - Avg: {avg_query_time:.2f}ms, "
                  f"Max: {max_query_time:.2f}ms")
            
        finally:
            await manager.close()


@pytest.mark.performance
class TestPerformanceMonitoring:
    """Test performance monitoring system performance."""
    
    @pytest.mark.asyncio
    async def test_performance_monitor_overhead(self, temp_dir):
        """Test performance monitoring overhead."""
        monitor = PerformanceMonitor(data_dir=str(temp_dir))
        
        try:
            # Measure overhead of performance monitoring
            def dummy_operation():
                # Simulate some work
                time.sleep(0.001)  # 1ms
                return sum(range(1000))
            
            # Measure without monitoring
            start_time = time.time()
            for _ in range(100):
                dummy_operation()
            unmonitored_time = time.time() - start_time
            
            # Measure with monitoring using PerformanceTimer
            start_time = time.time()
            for _ in range(100):
                with PerformanceTimer(monitor, ComponentType.AGENT, "dummy_op"):
                    dummy_operation()
            monitored_time = time.time() - start_time
            
            # Calculate overhead
            overhead = monitored_time - unmonitored_time
            overhead_percentage = (overhead / unmonitored_time) * 100
            
            # Monitoring overhead should be minimal (< 10%)
            assert overhead_percentage < 10
            
            print(f"Performance monitoring overhead: "
                  f"{overhead_percentage:.2f}%")
            
        finally:
            # No cleanup method needed
            pass
    
    @pytest.mark.asyncio
    async def test_metrics_collection_performance(self, temp_dir):
        """Test metrics collection performance."""
        monitor = PerformanceMonitor(data_dir=str(temp_dir))
        
        try:
            # Generate many metrics
            start_time = time.time()
            
            for i in range(1000):
                with PerformanceTimer(monitor, ComponentType.AGENT,
                                    f"operation_{i % 10}"):
                    time.sleep(0.0001)  # 0.1ms
            
            collection_time = time.time() - start_time
            
            # Get metrics using the correct method
            metrics_start = time.time()
            metrics = monitor.get_recent_metrics()
            metrics_time = (time.time() - metrics_start) * 1000
            
            # Metrics collection should be fast
            assert metrics_time < 100  # < 100ms
            assert len(metrics) > 0
            
            print(f"Collected {len(metrics)} metrics in "
                  f"{collection_time:.2f}s")
            print(f"Metrics retrieval time: {metrics_time:.2f}ms")
            
        finally:
            # No cleanup method needed
            pass


@pytest.mark.performance
@pytest.mark.slow
class TestStressTests:
    """Stress tests for system limits."""
    
    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(self):
        """Test concurrent tool execution performance."""
        registry = ToolRegistry()
        
        from tests.unit.test_tools_registry import MockCalculatorTool
        tool = MockCalculatorTool()
        registry.register(tool)
        
        # Execute tools concurrently
        async def execute_tool():
            return await registry.execute_tool("calculator", expression="2+2")
        
        start_time = time.time()
        
        # Run 50 concurrent executions
        tasks = [execute_tool() for _ in range(50)]
        results = await asyncio.gather(*tasks)
        
        execution_time = time.time() - start_time
        
        # All executions should succeed
        for result in results:
            assert result.success is True
        
        # Concurrent execution should be efficient
        assert execution_time < 5.0  # < 5 seconds for 50 concurrent executions
        
        print(f"50 concurrent tool executions completed in {execution_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_large_session_performance(self, temp_dir):
        """Test performance with large sessions."""
        manager = SessionManager(data_dir=str(temp_dir))
        await manager.initialize()
        
        try:
            # Create session with many messages
            session = await manager.start_new_session("Large Session Test")
            
            start_time = time.time()
            
            # Add 1000 messages
            for i in range(1000):
                await manager.add_message_to_current_session(
                    f"Large session message {i} with some content",
                    MessageType.USER if i % 2 == 0 else MessageType.ASSISTANT
                )
            
            insertion_time = time.time() - start_time
            
            # Query the large session
            query_start = time.time()
            large_session = await manager.get_session(session.id)
            query_time = time.time() - query_start
            
            assert large_session is not None
            assert len(large_session.messages) == 1000
            
            # Performance should still be reasonable
            assert insertion_time < 30.0  # < 30 seconds for 1000 insertions
            assert query_time < 5.0  # < 5 seconds to query large session
            
            print(f"1000 message insertion time: {insertion_time:.2f}s")
            print(f"Large session query time: {query_time:.2f}s")
            
        finally:
            await manager.close()
    
    def test_memory_leak_detection(self, mock_audio_data):
        """Test for memory leaks in audio processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process audio repeatedly
        vad = VoiceActivityDetector()
        
        for iteration in range(10):
            # Process 10 seconds of audio
            chunk_size = 1600
            for _ in range(100):  # 10 seconds worth
                for i in range(0, len(mock_audio_data), chunk_size):
                    chunk = mock_audio_data[i:i + chunk_size]
                    if len(chunk) == chunk_size:
                        vad.process_audio(chunk)
            
            # Check memory every iteration
            current_memory = process.memory_info().rss
            memory_growth = current_memory - initial_memory
            
            # Memory growth should be bounded
            max_allowed_growth = 50 * 1024 * 1024  # 50MB
            if memory_growth > max_allowed_growth:
                pytest.fail(f"Memory leak detected: {memory_growth / 1024 / 1024:.2f}MB growth")
        
        final_memory = process.memory_info().rss
        total_growth = final_memory - initial_memory
        
        print(f"Total memory growth after stress test: {total_growth / 1024 / 1024:.2f}MB")


@pytest.mark.performance
class TestRegressionBenchmarks:
    """Regression benchmarks to track performance over time."""
    
    def test_baseline_vad_performance(self, mock_audio_data):
        """Baseline VAD performance benchmark."""
        vad = VoiceActivityDetector()
        
        # Process 10 seconds of audio
        iterations = 100
        chunk_size = 1600
        
        start_time = time.time()
        
        for _ in range(iterations):
            for i in range(0, len(mock_audio_data), chunk_size):
                chunk = mock_audio_data[i:i + chunk_size]
                if len(chunk) == chunk_size:
                    vad.process_audio(chunk)
        
        total_time = time.time() - start_time
        throughput = (iterations * len(mock_audio_data)) / total_time / 16000  # seconds of audio per second
        
        # Log baseline performance
        print(f"VAD Baseline: {throughput:.2f}x real-time processing")
        
        # Should process faster than real-time
        assert throughput > 1.0
    
    @pytest.mark.asyncio
    async def test_baseline_tool_performance(self):
        """Baseline tool execution performance benchmark."""
        registry = ToolRegistry()
        
        from tests.unit.test_tools_registry import MockCalculatorTool
        tool = MockCalculatorTool()
        registry.register(tool)
        
        # Execute tool many times
        iterations = 1000
        start_time = time.time()
        
        for _ in range(iterations):
            result = await registry.execute_tool("calculator", expression="2+2")
            assert result.success is True
        
        total_time = time.time() - start_time
        throughput = iterations / total_time
        
        # Log baseline performance
        print(f"Tool Execution Baseline: {throughput:.2f} executions/second")
        
        # Should execute many tools per second
        assert throughput > 100