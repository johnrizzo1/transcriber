"""
Integration helpers for adding performance monitoring to pipeline components.
"""

import asyncio
import logging
from typing import Optional, Any, Dict

from .monitor import PerformanceMonitor, ComponentType, PerformanceTimer
from .profiler import ProfilerManager, ProfilerContext
from .benchmarks import BenchmarkSuite
from .optimizer import ResourceOptimizer

logger = logging.getLogger(__name__)


class PerformanceIntegration:
    """
    Helper class for integrating performance monitoring into pipeline components.
    
    This class provides a unified interface for adding performance monitoring,
    profiling, and optimization to existing pipeline components.
    """
    
    def __init__(self, data_dir: str = "./data", enable_all: bool = True):
        self.data_dir = data_dir
        
        # Initialize performance components
        self.monitor = PerformanceMonitor(data_dir, enable_storage=enable_all)
        self.profiler = ProfilerManager(data_dir) if enable_all else None
        self.benchmarks = BenchmarkSuite(data_dir) if enable_all else None
        self.optimizer = ResourceOptimizer(data_dir) if enable_all else None
        
        # Cross-reference components
        if self.benchmarks:
            self.benchmarks.set_performance_monitor(self.monitor)
        if self.optimizer:
            self.optimizer.set_performance_monitor(self.monitor)
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize performance monitoring."""
        if self._initialized:
            return
        
        logger.info("Initializing performance monitoring integration")
        
        # Start monitoring if enabled
        await self.monitor.start_monitoring()
        
        self._initialized = True
        logger.info("Performance monitoring integration initialized")
    
    async def cleanup(self) -> None:
        """Cleanup performance monitoring."""
        if not self._initialized:
            return
        
        logger.info("Cleaning up performance monitoring")
        
        # Stop monitoring
        await self.monitor.stop_monitoring()
        
        self._initialized = False
    
    def time_operation(
        self, 
        component: ComponentType, 
        operation: str = "default"
    ) -> PerformanceTimer:
        """Get a performance timer for an operation."""
        return PerformanceTimer(self.monitor, component, operation)
    
    def profile_operation(
        self,
        component: ComponentType,
        operation: str = "default",
        enable_memory: bool = False
    ) -> Optional[ProfilerContext]:
        """Get a profiler context for an operation."""
        if not self.profiler:
            return None
        
        return ProfilerContext(
            self.profiler, component, operation, enable_memory
        )
    
    async def run_component_benchmark(
        self,
        component: ComponentType,
        benchmark_func,
        iterations: int = 100
    ) -> Optional[Any]:
        """Run a benchmark for a component."""
        if not self.benchmarks:
            return None
        
        return await self.benchmarks.run_benchmark(
            name=f"{component.value}_benchmark",
            component=component,
            benchmark_func=benchmark_func,
            iterations=iterations
        )
    
    async def optimize_component(
        self,
        component: ComponentType,
        current_settings: Dict[str, Any]
    ) -> Optional[Any]:
        """Optimize a component's settings."""
        if not self.optimizer:
            return None
        
        if component == ComponentType.AUDIO_CAPTURE:
            return await self.optimizer.optimize_audio_processing(current_settings)
        elif component == ComponentType.PIPELINE:
            return await self.optimizer.optimize_memory_usage()
        else:
            logger.warning(f"No optimization available for component: {component}")
            return None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        summary = self.monitor.get_performance_summary()
        
        # Add optimization recommendations if available
        if self.optimizer:
            summary["optimization_recommendations"] = (
                self.optimizer.get_optimization_recommendations()
            )
        
        # Add recent benchmark results if available
        if self.benchmarks:
            recent_benchmarks = self.benchmarks.get_benchmark_history(days=1)
            summary["recent_benchmarks"] = len(recent_benchmarks)
        
        return summary


# Decorator for easy performance monitoring
def monitor_component_performance(
    component: ComponentType,
    operation: str = "default",
    enable_profiling: bool = False
):
    """
    Decorator to automatically add performance monitoring to a function.
    
    Usage:
        @monitor_component_performance(ComponentType.STT, "transcribe")
        async def transcribe_audio(self, audio_data):
            # Function implementation
            pass
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Try to get performance integration from first argument
            perf_integration = None
            if args and hasattr(args[0], 'performance_integration'):
                perf_integration = args[0].performance_integration
            
            if perf_integration and isinstance(perf_integration, PerformanceIntegration):
                # Use performance monitoring
                with perf_integration.time_operation(component, operation):
                    if enable_profiling:
                        profiler_ctx = perf_integration.profile_operation(
                            component, operation
                        )
                        if profiler_ctx:
                            with profiler_ctx:
                                return await func(*args, **kwargs)
                    
                    return await func(*args, **kwargs)
            else:
                # No performance monitoring available
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            # Try to get performance integration from first argument
            perf_integration = None
            if args and hasattr(args[0], 'performance_integration'):
                perf_integration = args[0].performance_integration
            
            if perf_integration and isinstance(perf_integration, PerformanceIntegration):
                # Use performance monitoring
                with perf_integration.time_operation(component, operation):
                    if enable_profiling:
                        profiler_ctx = perf_integration.profile_operation(
                            component, operation
                        )
                        if profiler_ctx:
                            with profiler_ctx:
                                return func(*args, **kwargs)
                    
                    return func(*args, **kwargs)
            else:
                # No performance monitoring available
                return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Example usage functions
async def example_integration():
    """Example of how to integrate performance monitoring."""
    
    # Initialize performance integration
    perf = PerformanceIntegration()
    await perf.initialize()
    
    try:
        # Example: Time an operation
        with perf.time_operation(ComponentType.STT, "transcribe"):
            # Simulate STT processing
            await asyncio.sleep(0.1)
        
        # Example: Profile an operation
        profiler_ctx = perf.profile_operation(ComponentType.LLM, "generate")
        if profiler_ctx:
            with profiler_ctx:
                # Simulate LLM processing
                await asyncio.sleep(0.2)
        
        # Example: Run a benchmark
        async def dummy_benchmark():
            await asyncio.sleep(0.05)
        
        benchmark_result = await perf.run_component_benchmark(
            ComponentType.TTS,
            dummy_benchmark,
            iterations=10
        )
        
        if benchmark_result:
            print(f"Benchmark result: {benchmark_result.avg_time_ms:.2f}ms average")
        
        # Example: Get performance summary
        summary = perf.get_performance_summary()
        print(f"Performance summary: {summary}")
        
    finally:
        await perf.cleanup()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_integration())