"""
Test script for the Performance Optimization System.
"""

import asyncio
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add the transcriber package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from transcriber.performance.integration import PerformanceIntegration
from transcriber.performance.monitor import ComponentType


async def test_performance_system():
    """Test the complete performance optimization system."""
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Initialize performance integration
        print("\n1. Initializing Performance Integration...")
        perf = PerformanceIntegration(data_dir=temp_dir, enable_all=True)
        await perf.initialize()
        print("✓ Performance integration initialized")
        
        # Test performance monitoring
        print("\n2. Testing Performance Monitoring...")
        with perf.time_operation(ComponentType.STT, "test_transcribe"):
            await asyncio.sleep(0.1)  # Simulate processing
        print("✓ Performance timing completed")
        
        # Test profiling
        print("\n3. Testing Profiling...")
        profiler_ctx = perf.profile_operation(
            ComponentType.LLM, 
            "test_generate",
            enable_memory=True
        )
        if profiler_ctx:
            with profiler_ctx:
                # Simulate some work
                data = [i**2 for i in range(1000)]
                await asyncio.sleep(0.05)
            print("✓ Profiling completed")
        else:
            print("⚠ Profiling not available")
        
        # Test benchmarking
        print("\n4. Testing Benchmarking...")
        
        async def test_benchmark():
            """Simple benchmark function."""
            await asyncio.sleep(0.02)
            return sum(range(100))
        
        benchmark_result = await perf.run_component_benchmark(
            ComponentType.TTS,
            test_benchmark,
            iterations=5
        )
        
        if benchmark_result:
            print(f"✓ Benchmark completed: {benchmark_result.avg_time_ms:.2f}ms avg")
        else:
            print("⚠ Benchmarking not available")
        
        # Test optimization
        print("\n5. Testing Optimization...")
        audio_settings = {
            'sample_rate': 16000,
            'chunk_size': 1024,
            'channels': 1
        }
        
        optimization_result = await perf.optimize_component(
            ComponentType.AUDIO_CAPTURE,
            audio_settings
        )
        
        if optimization_result:
            print("✓ Audio optimization completed")
        else:
            print("⚠ Optimization not available")
        
        # Test performance summary
        print("\n6. Getting Performance Summary...")
        summary = perf.get_performance_summary()
        print(f"✓ Performance summary retrieved:")
        print(f"  - Components monitored: {len(summary.get('components', {}))}")
        print(f"  - Total operations: {summary.get('total_operations', 0)}")
        
        # Test multiple operations
        print("\n7. Testing Multiple Operations...")
        operations = [
            (ComponentType.VAD, "voice_detection"),
            (ComponentType.STT, "transcription"),
            (ComponentType.LLM, "response_generation"),
            (ComponentType.TTS, "speech_synthesis"),
        ]
        
        for component, operation in operations:
            with perf.time_operation(component, operation):
                await asyncio.sleep(0.01)  # Quick operation
        
        print("✓ Multiple operations completed")
        
        # Final summary
        print("\n8. Final Performance Summary...")
        final_summary = perf.get_performance_summary()
        print(f"✓ Final summary:")
        for component, stats in final_summary.get('components', {}).items():
            if stats.get('operation_count', 0) > 0:
                avg_time = stats.get('avg_time_ms', 0)
                count = stats.get('operation_count', 0)
                print(f"  - {component}: {count} ops, {avg_time:.2f}ms avg")
        
        print("\n✅ All performance system tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        try:
            await perf.cleanup()
            print("\n🧹 Performance integration cleaned up")
        except:
            pass
        
        # Remove temporary directory
        try:
            shutil.rmtree(temp_dir)
            print(f"🗑️ Temporary directory removed: {temp_dir}")
        except:
            print(f"⚠ Could not remove temporary directory: {temp_dir}")


async def test_decorator_integration():
    """Test the performance monitoring decorator."""
    
    print("\n" + "="*50)
    print("TESTING DECORATOR INTEGRATION")
    print("="*50)
    
    # Create a mock class with performance integration
    class MockComponent:
        def __init__(self, temp_dir):
            self.performance_integration = PerformanceIntegration(
                data_dir=temp_dir, 
                enable_all=True
            )
        
        async def initialize(self):
            await self.performance_integration.initialize()
        
        async def cleanup(self):
            await self.performance_integration.cleanup()
        
        # Import the decorator
        from transcriber.performance.integration import monitor_component_performance
        
        @monitor_component_performance(ComponentType.STT, "mock_transcribe")
        async def mock_transcribe(self, audio_data):
            """Mock transcription function."""
            await asyncio.sleep(0.05)
            return f"Transcribed: {len(audio_data)} bytes"
        
        @monitor_component_performance(ComponentType.LLM, "mock_generate", True)
        async def mock_generate(self, prompt):
            """Mock generation function with profiling."""
            await asyncio.sleep(0.03)
            return f"Generated response for: {prompt[:20]}..."
    
    # Test the decorator
    temp_dir = tempfile.mkdtemp()
    
    try:
        component = MockComponent(temp_dir)
        await component.initialize()
        
        print("\n1. Testing decorated transcription...")
        result1 = await component.mock_transcribe(b"fake audio data")
        print(f"✓ Transcription result: {result1}")
        
        print("\n2. Testing decorated generation with profiling...")
        result2 = await component.mock_generate("What is the weather like?")
        print(f"✓ Generation result: {result2}")
        
        print("\n3. Getting decorator performance summary...")
        summary = component.performance_integration.get_performance_summary()
        print("✓ Decorator performance summary:")
        for comp, stats in summary.get('components', {}).items():
            if stats.get('operation_count', 0) > 0:
                print(f"  - {comp}: {stats['operation_count']} operations")
        
        print("\n✅ Decorator integration tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Decorator test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        try:
            await component.cleanup()
            shutil.rmtree(temp_dir)
        except:
            pass


async def main():
    """Run all performance system tests."""
    print("🚀 Starting Performance Optimization System Tests")
    print("="*60)
    
    # Test main performance system
    await test_performance_system()
    
    # Test decorator integration
    await test_decorator_integration()
    
    print("\n" + "="*60)
    print("🎉 All Performance System Tests Completed!")


if __name__ == "__main__":
    asyncio.run(main())