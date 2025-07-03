# Performance Optimization System

The AI Voice Agent includes a comprehensive performance optimization system that provides monitoring, profiling, benchmarking, and optimization capabilities for all pipeline components.

## Overview

The performance system consists of four main components:

1. **Performance Monitor** - Real-time metrics collection and monitoring
2. **Profiler** - Detailed code profiling and bottleneck identification
3. **Benchmarks** - Automated performance testing and regression detection
4. **Optimizer** - Resource optimization and performance tuning

## Quick Start

### CLI Commands

The performance system provides three main CLI commands:

```bash
# View current performance metrics
poetry run python -m transcriber performance

# Run benchmarks on pipeline components
poetry run python -m transcriber benchmark --component stt --iterations 50

# Profile the system for bottlenecks
poetry run python -m transcriber profile --duration 30 --memory
```

### Integration in Code

```python
from transcriber.performance.integration import PerformanceIntegration
from transcriber.performance.monitor import ComponentType

# Initialize performance monitoring
perf = PerformanceIntegration()
await perf.initialize()

# Time an operation
with perf.time_operation(ComponentType.STT, "transcribe"):
    # Your STT code here
    result = await transcribe_audio(audio_data)

# Profile an operation
profiler_ctx = perf.profile_operation(ComponentType.LLM, "generate")
if profiler_ctx:
    with profiler_ctx:
        # Your LLM code here
        response = await generate_response(prompt)

# Run a benchmark
async def benchmark_function():
    # Your benchmark code
    await some_operation()

result = await perf.run_component_benchmark(
    ComponentType.TTS,
    benchmark_function,
    iterations=100
)

# Get performance summary
summary = perf.get_performance_summary()
print(f"Average response time: {summary['avg_response_time']}ms")

# Cleanup
await perf.cleanup()
```

## Components

### 1. Performance Monitor

The Performance Monitor collects real-time metrics for all pipeline components:

- **Latency tracking** - Response times for each component
- **Resource usage** - CPU, memory, and system metrics
- **Operation counting** - Number of operations per component
- **Error tracking** - Failed operations and error rates

#### Usage

```python
from transcriber.performance.monitor import PerformanceMonitor, ComponentType

monitor = PerformanceMonitor("./data")
await monitor.start_monitoring()

# Time an operation
with monitor.time_operation(ComponentType.STT):
    # Your code here
    pass

# Get metrics
metrics = monitor.get_current_metrics()
summary = monitor.get_performance_summary()

await monitor.stop_monitoring()
```

#### CLI Usage

```bash
# View current performance metrics
poetry run python -m transcriber performance

# Show live metrics (updates every 2 seconds)
poetry run python -m transcriber performance --live

# Filter by component
poetry run python -m transcriber performance --component stt

# Show last 30 minutes of data
poetry run python -m transcriber performance --minutes 30

# Export metrics to file
poetry run python -m transcriber performance --export metrics.json
```

### 2. Profiler

The Profiler provides detailed code analysis to identify performance bottlenecks:

- **CPU profiling** - Function call analysis and timing
- **Memory profiling** - Memory usage tracking and leak detection
- **Call graph analysis** - Function call relationships
- **Hot spot identification** - Most time-consuming functions

#### Usage

```python
from transcriber.performance.profiler import ProfilerManager, ProfilerContext

profiler = ProfilerManager("./data")

# Profile a code block
with ProfilerContext(profiler, ComponentType.LLM, "generate", enable_memory=True):
    # Your code here
    result = await expensive_operation()

# Get profiling results
results = profiler.get_profile_results()
for result in results:
    print(f"Operation: {result.operation}")
    print(f"Duration: {result.duration_ms}ms")
    print(f"Top functions: {result.top_functions[:5]}")
```

#### CLI Usage

```bash
# Profile for 60 seconds (default)
poetry run python -m transcriber profile

# Profile specific component
poetry run python -m transcriber profile --component stt --duration 30

# Enable memory profiling
poetry run python -m transcriber profile --memory --duration 45

# Save results to file
poetry run python -m transcriber profile --output profile_results.json
```

### 3. Benchmarks

The Benchmark system provides automated performance testing:

- **Component benchmarks** - Individual component performance
- **Pipeline benchmarks** - End-to-end performance testing
- **Regression detection** - Performance degradation alerts
- **Historical tracking** - Performance trends over time

#### Usage

```python
from transcriber.performance.benchmarks import BenchmarkSuite

benchmarks = BenchmarkSuite("./data")

# Define a benchmark function
async def stt_benchmark():
    # Simulate STT processing
    await transcribe_audio(test_audio)

# Run benchmark
result = await benchmarks.run_benchmark(
    name="stt_performance",
    component=ComponentType.STT,
    benchmark_func=stt_benchmark,
    iterations=100
)

print(f"Average time: {result.avg_time_ms}ms")
print(f"95th percentile: {result.p95_time_ms}ms")

# Get benchmark history
history = benchmarks.get_benchmark_history(days=7)
```

#### CLI Usage

```bash
# Run all benchmarks
poetry run python -m transcriber benchmark

# Benchmark specific component
poetry run python -m transcriber benchmark --component stt --iterations 50

# Save results to CSV
poetry run python -m transcriber benchmark --output results.csv --format csv

# Benchmark the entire pipeline
poetry run python -m transcriber benchmark --component pipeline --iterations 10
```

### 4. Optimizer

The Optimizer provides automated performance tuning:

- **Audio processing optimization** - Sample rate and buffer size tuning
- **Memory optimization** - Memory usage reduction
- **Model caching** - Efficient model loading and caching
- **Resource allocation** - Optimal resource distribution

#### Usage

```python
from transcriber.performance.optimizer import ResourceOptimizer

optimizer = ResourceOptimizer("./data")

# Optimize audio settings
current_settings = {
    'sample_rate': 16000,
    'chunk_size': 1024,
    'channels': 1
}

optimized = await optimizer.optimize_audio_processing(current_settings)
print(f"Optimized settings: {optimized}")

# Optimize memory usage
memory_stats = await optimizer.optimize_memory_usage()
print(f"Memory freed: {memory_stats['freed_mb']}MB")

# Get optimization recommendations
recommendations = optimizer.get_optimization_recommendations()
for rec in recommendations:
    print(f"- {rec['description']}")
    print(f"  Expected improvement: {rec['expected_improvement']}")
```

## Decorator Integration

For easy integration, use the performance monitoring decorator:

```python
from transcriber.performance.integration import monitor_component_performance
from transcriber.performance.monitor import ComponentType

class VoiceAgent:
    def __init__(self):
        # Initialize with performance integration
        self.performance_integration = PerformanceIntegration()
    
    @monitor_component_performance(ComponentType.STT, "transcribe")
    async def transcribe_audio(self, audio_data):
        # Your transcription code
        return transcribed_text
    
    @monitor_component_performance(ComponentType.LLM, "generate", enable_profiling=True)
    async def generate_response(self, prompt):
        # Your LLM code with profiling enabled
        return response
```

## Configuration

Performance monitoring can be configured through the main configuration system:

```python
# In transcriber/config.py
class PerformanceConfig(BaseSettings):
    # Monitoring settings
    enable_monitoring: bool = True
    monitoring_interval: float = 1.0
    max_metrics_history: int = 10000
    
    # Profiling settings
    enable_profiling: bool = False
    enable_memory_profiling: bool = False
    profiling_sample_rate: float = 0.1
    
    # Benchmarking settings
    default_benchmark_iterations: int = 100
    benchmark_timeout: float = 300.0
    
    # Optimization settings
    enable_auto_optimization: bool = False
    optimization_interval: float = 3600.0  # 1 hour
```

## Data Storage

Performance data is stored in SQLite databases in the configured data directory:

- `performance_metrics.db` - Real-time metrics and monitoring data
- `profile_results.db` - Profiling results and analysis
- `benchmark_results.db` - Benchmark results and history
- `optimization_logs.db` - Optimization actions and results

## Best Practices

### 1. Monitoring

- Enable monitoring in production for continuous visibility
- Use component-specific filtering to focus on problem areas
- Export metrics regularly for historical analysis
- Set up alerts for performance degradation

### 2. Profiling

- Profile during development and testing phases
- Use memory profiling to identify memory leaks
- Focus profiling on specific components showing issues
- Profile under realistic load conditions

### 3. Benchmarking

- Run benchmarks before and after code changes
- Use consistent test data for reliable comparisons
- Benchmark individual components and the full pipeline
- Track benchmark results over time

### 4. Optimization

- Start with profiling to identify bottlenecks
- Test optimizations with benchmarks
- Monitor the impact of optimizations in production
- Document optimization changes and their effects

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```bash
   # Profile memory usage
   poetry run python -m transcriber profile --memory --duration 60
   
   # Check for memory leaks
   poetry run python -m transcriber performance --component memory
   ```

2. **Slow Response Times**
   ```bash
   # Identify slow components
   poetry run python -m transcriber performance --live
   
   # Profile the slowest component
   poetry run python -m transcriber profile --component stt
   ```

3. **Performance Regression**
   ```bash
   # Compare current vs historical benchmarks
   poetry run python -m transcriber benchmark --component pipeline
   
   # Check recent performance trends
   poetry run python -m transcriber performance --minutes 60
   ```

### Performance Tuning Tips

1. **Audio Processing**
   - Optimize sample rate for your use case
   - Adjust chunk size for latency vs throughput
   - Use appropriate VAD thresholds

2. **STT Performance**
   - Choose the right Whisper model size
   - Consider using faster-whisper for better performance
   - Optimize audio preprocessing

3. **LLM Performance**
   - Use appropriate model sizes
   - Implement response caching
   - Optimize prompt engineering

4. **TTS Performance**
   - Cache frequently used phrases
   - Optimize voice settings
   - Use streaming for long responses

## Integration Examples

### Basic Integration

```python
async def main():
    # Initialize performance monitoring
    perf = PerformanceIntegration()
    await perf.initialize()
    
    try:
        # Your application code with monitoring
        with perf.time_operation(ComponentType.PIPELINE, "full_cycle"):
            result = await process_voice_input(audio_data)
        
        # Get performance summary
        summary = perf.get_performance_summary()
        logger.info(f"Performance: {summary}")
        
    finally:
        await perf.cleanup()
```

### Advanced Integration

```python
class PerformanceAwareVoiceAgent:
    def __init__(self):
        self.performance = PerformanceIntegration(enable_all=True)
        self.optimization_due = False
    
    async def initialize(self):
        await self.performance.initialize()
    
    async def process_audio(self, audio_data):
        # Monitor the full pipeline
        with self.performance.time_operation(ComponentType.PIPELINE, "process"):
            # Individual component monitoring
            with self.performance.time_operation(ComponentType.VAD, "detect"):
                voice_detected = await self.detect_voice(audio_data)
            
            if voice_detected:
                with self.performance.time_operation(ComponentType.STT, "transcribe"):
                    text = await self.transcribe(audio_data)
                
                with self.performance.time_operation(ComponentType.LLM, "generate"):
                    response = await self.generate_response(text)
                
                with self.performance.time_operation(ComponentType.TTS, "synthesize"):
                    audio_response = await self.synthesize_speech(response)
                
                return audio_response
        
        # Check if optimization is needed
        if self.should_optimize():
            await self.run_optimization()
    
    async def should_optimize(self):
        summary = self.performance.get_performance_summary()
        # Check if any component is performing poorly
        for component, stats in summary.get('components', {}).items():
            if stats.get('avg_time_ms', 0) > self.get_threshold(component):
                return True
        return False
    
    async def run_optimization(self):
        # Run component-specific optimizations
        for component in [ComponentType.AUDIO_CAPTURE, ComponentType.STT, ComponentType.LLM]:
            current_settings = self.get_component_settings(component)
            optimized = await self.performance.optimize_component(component, current_settings)
            if optimized:
                self.apply_optimized_settings(component, optimized)
```

This performance optimization system provides comprehensive monitoring and optimization capabilities for the AI Voice Agent, enabling you to maintain optimal performance and quickly identify and resolve performance issues.