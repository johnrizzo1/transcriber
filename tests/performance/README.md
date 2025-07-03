# Performance Testing Suite

This directory contains comprehensive performance tests for the AI Voice Agent system. The test suite is designed to ensure the system meets performance requirements and to track performance regressions over time.

## Test Structure

### Test Categories

- **Unit Performance Tests** (`test_benchmarks.py`): Individual component performance
- **Stress Tests** (`test_stress.py`): System behavior under load
- **Integration Performance** (`test_voice_pipeline_fixed.py`): End-to-end performance
- **Configuration** (`test_config.py`): Test utilities and configuration

### Performance Markers

Tests are organized using pytest markers:

- `@pytest.mark.performance`: General performance tests
- `@pytest.mark.stress`: Stress and load tests
- `@pytest.mark.regression`: Regression tracking tests
- `@pytest.mark.slow`: Long-running tests

## Running Performance Tests

### Quick Start

```bash
# Run all performance tests
python tests/run_performance_tests.py

# Run specific test types
python tests/run_performance_tests.py --type stress
python tests/run_performance_tests.py --type regression

# Verbose output with detailed reporting
python tests/run_performance_tests.py --verbose
```

### Using pytest directly

```bash
# Run all performance tests
pytest -m performance tests/performance/

# Run stress tests only
pytest -m stress tests/performance/

# Run with coverage
pytest -m performance --cov=transcriber tests/performance/
```

## Performance Requirements

### Audio Processing
- **VAD Processing**: < 50ms for 1 second of audio
- **Real-time Factor**: > 1.0x (faster than real-time)
- **Memory Growth**: < 100MB during extended processing

### Tool Execution
- **Average Execution**: < 10ms per tool
- **Maximum Execution**: < 50ms per tool
- **Throughput**: > 100 tools/second

### Session Management
- **Session Creation**: < 100ms
- **Message Insertion**: < 50ms per message
- **Session Query**: < 100ms
- **Message Throughput**: > 50 messages/second

### System Integration
- **End-to-end Latency**: < 500ms
- **Concurrent Sessions**: Support 5+ concurrent sessions
- **Memory Stability**: < 100MB growth under sustained load

## Test Components

### Audio Performance Tests

```python
class TestAudioPerformance:
    def test_vad_processing_speed(self):
        """Test VAD processing meets speed requirements"""
    
    def test_vad_memory_usage(self):
        """Test VAD memory usage remains stable"""
```

### Tool Performance Tests

```python
class TestToolPerformance:
    def test_tool_registry_lookup_speed(self):
        """Test tool lookup performance"""
    
    def test_tool_execution_performance(self):
        """Test tool execution speed"""
```

### Session Performance Tests

```python
class TestSessionPerformance:
    def test_session_creation_speed(self):
        """Test session creation performance"""
    
    def test_message_insertion_speed(self):
        """Test message insertion performance"""
    
    def test_session_query_performance(self):
        """Test session query performance"""
```

### Stress Tests

```python
class TestStressTests:
    def test_vad_continuous_processing(self):
        """Test VAD under continuous load"""
    
    def test_concurrent_tool_execution(self):
        """Test concurrent tool execution"""
    
    def test_session_high_volume(self):
        """Test high-volume message processing"""
    
    def test_memory_stability_under_load(self):
        """Test memory stability under sustained load"""
```

## Performance Monitoring

### Baseline Tracking

Performance baselines are stored in `performance_baseline.txt`:

```
VAD_BASELINE: 2.5x
TOOL_BASELINE: 150.0 exec/sec
SESSION_BASELINE: 75.0 msgs/sec
```

### Regression Detection

Tests automatically compare current performance against baselines:

- **Green**: Performance within 10% of baseline
- **Yellow**: Performance degraded by 10-20%
- **Red**: Performance degraded by >20% (test failure)

### Performance Reports

Detailed reports are generated in `performance_report.txt`:

```
Performance Test Report
=======================

VAD Processing:
  Count: 100
  Average: 25.5 ms
  Min: 20.1 ms
  Max: 35.2 ms
  ✓ Stable: +2.1% vs baseline

Tool Execution:
  Count: 1000
  Average: 8.5 ms
  Min: 5.2 ms
  Max: 15.8 ms
  ✅ IMPROVEMENT: +12.5% vs baseline
```

## Configuration

### Performance Thresholds

Thresholds are configurable in `test_config.py`:

```python
@pytest.fixture
def performance_thresholds():
    return {
        'vad_processing': 50.0,  # milliseconds
        'tool_execution_avg': 10.0,
        'memory_growth_limit': 100.0,  # MB
        # ... more thresholds
    }
```

### Test Configuration

```python
@pytest.fixture
def performance_config():
    return {
        'stress_test_duration': 5.0,  # seconds
        'load_test_iterations': 100,
        'large_session_messages': 1000,
        # ... more config
    }
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Run Performance Tests
  run: |
    python tests/run_performance_tests.py --type regression
    
- name: Upload Performance Report
  uses: actions/upload-artifact@v3
  with:
    name: performance-report
    path: performance_report.txt
```

### Performance Monitoring

- Baseline updates on major releases
- Regression alerts on performance degradation
- Trend tracking over time

## Troubleshooting

### Common Issues

1. **Tests Running Slowly**
   - Use `--quick` flag to skip slow tests
   - Run specific test categories instead of all tests

2. **Memory Issues**
   - Monitor system memory during tests
   - Adjust batch sizes in configuration

3. **Flaky Performance Tests**
   - Run tests multiple times for stability
   - Adjust thresholds if system-dependent

### Debug Mode

```bash
# Run with detailed output
python tests/run_performance_tests.py --verbose

# Run single test for debugging
pytest -v -s tests/performance/test_benchmarks.py::TestAudioPerformance::test_vad_processing_speed
```

## Contributing

### Adding New Performance Tests

1. Create test in appropriate file (`test_benchmarks.py`, `test_stress.py`, etc.)
2. Add appropriate markers (`@pytest.mark.performance`, etc.)
3. Follow naming convention: `test_<component>_<metric>`
4. Include performance assertions with clear thresholds
5. Add documentation for new requirements

### Updating Baselines

```bash
# Run tests to generate new baselines
python tests/run_performance_tests.py --type regression

# Review and commit updated baseline file
git add performance_baseline.txt
git commit -m "Update performance baselines"
```

## Performance Test Results

Current performance status:

- ✅ **Audio Processing**: Meeting all latency requirements
- ✅ **Tool Execution**: Exceeding throughput targets  
- ✅ **Session Management**: Stable performance under load
- ✅ **Memory Usage**: No memory leaks detected
- ✅ **Integration**: End-to-end latency within limits

Last updated: [Generated automatically by test runner]