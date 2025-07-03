"""
Performance test configuration and utilities.
"""

import pytest
from typing import Dict, Any


@pytest.fixture
def performance_thresholds() -> Dict[str, float]:
    """Performance thresholds for various components."""
    return {
        # Audio processing thresholds (milliseconds)
        'vad_processing': 50.0,  # VAD should process 1s audio in <50ms
        'audio_latency': 100.0,  # Total audio pipeline latency
        
        # Tool execution thresholds
        'tool_execution_avg': 10.0,  # Average tool execution time
        'tool_execution_max': 50.0,  # Maximum tool execution time
        
        # Session management thresholds
        'session_creation': 100.0,  # Session creation time
        'message_insertion': 50.0,  # Message insertion time
        'session_query': 100.0,  # Session query time
        
        # Memory thresholds (MB)
        'memory_growth_limit': 100.0,  # Maximum memory growth
        'memory_leak_threshold': 10.0,  # Memory leak detection
        
        # Throughput thresholds
        'vad_throughput': 1.0,  # Real-time factor for VAD
        'tool_throughput': 100.0,  # Tools per second
        'message_throughput': 50.0,  # Messages per second
    }


@pytest.fixture
def performance_config() -> Dict[str, Any]:
    """Configuration for performance tests."""
    return {
        # Test durations
        'stress_test_duration': 5.0,  # seconds
        'load_test_iterations': 100,
        'memory_test_iterations': 50,
        
        # Batch sizes for concurrent tests
        'small_batch': 10,
        'medium_batch': 25,
        'large_batch': 50,
        'stress_batch': 100,
        
        # Data sizes
        'small_session_messages': 20,
        'large_session_messages': 1000,
        'stress_session_count': 50,
        
        # Monitoring intervals
        'resource_monitor_interval': 1.0,  # seconds
        'performance_log_interval': 10,  # iterations
        
        # Regression tracking
        'baseline_file': 'performance_baseline.txt',
        'regression_tolerance': 0.1,  # 10% tolerance
    }


class PerformanceTracker:
    """Utility class for tracking performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.baselines = {}
    
    def record_metric(self, name: str, value: float, unit: str = ""):
        """Record a performance metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append({
            'value': value,
            'unit': unit,
            'timestamp': pytest.current_time if hasattr(pytest, 'current_time') else 0
        })
    
    def get_average(self, name: str) -> float:
        """Get average value for a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        
        values = [m['value'] for m in self.metrics[name]]
        return sum(values) / len(values)
    
    def get_max(self, name: str) -> float:
        """Get maximum value for a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        
        values = [m['value'] for m in self.metrics[name]]
        return max(values)
    
    def get_min(self, name: str) -> float:
        """Get minimum value for a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        
        values = [m['value'] for m in self.metrics[name]]
        return min(values)
    
    def load_baselines(self, filepath: str):
        """Load baseline metrics from file."""
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if ':' in line:
                        name, value_str = line.strip().split(':', 1)
                        try:
                            # Extract numeric value from string like "1.5x" or "100 exec/sec"
                            value = float(value_str.split()[0].rstrip('x'))
                            self.baselines[name] = value
                        except (ValueError, IndexError):
                            continue
        except FileNotFoundError:
            pass
    
    def check_regression(self, name: str, current_value: float, tolerance: float = 0.1) -> bool:
        """Check if current value represents a performance regression."""
        if name not in self.baselines:
            return False
        
        baseline = self.baselines[name]
        regression_threshold = baseline * (1 - tolerance)
        
        return current_value < regression_threshold
    
    def generate_report(self) -> str:
        """Generate a performance report."""
        lines = ["Performance Test Report", "=" * 25, ""]
        
        for name, measurements in self.metrics.items():
            if not measurements:
                continue
            
            values = [m['value'] for m in measurements]
            unit = measurements[0].get('unit', '')
            
            lines.append(f"{name}:")
            lines.append(f"  Count: {len(values)}")
            lines.append(f"  Average: {sum(values)/len(values):.2f} {unit}")
            lines.append(f"  Min: {min(values):.2f} {unit}")
            lines.append(f"  Max: {max(values):.2f} {unit}")
            
            # Check for regression if baseline exists
            if name in self.baselines:
                current_avg = sum(values) / len(values)
                baseline = self.baselines[name]
                change_pct = ((current_avg - baseline) / baseline) * 100
                
                if change_pct < -10:
                    lines.append(f"  ⚠️  REGRESSION: {change_pct:.1f}% vs baseline")
                elif change_pct > 10:
                    lines.append(f"  ✅ IMPROVEMENT: +{change_pct:.1f}% vs baseline")
                else:
                    lines.append(f"  ✓ Stable: {change_pct:+.1f}% vs baseline")
            
            lines.append("")
        
        return "\n".join(lines)


@pytest.fixture
def performance_tracker():
    """Fixture providing a performance tracker instance."""
    return PerformanceTracker()


# Performance test markers
pytest.mark.performance = pytest.mark.performance
pytest.mark.slow = pytest.mark.slow
pytest.mark.stress = pytest.mark.stress
pytest.mark.regression = pytest.mark.regression


def pytest_configure(config):
    """Configure pytest with performance test markers."""
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "stress: mark test as a stress test"
    )
    config.addinivalue_line(
        "markers", "regression: mark test as a regression test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle performance test markers."""
    # Skip performance tests by default unless explicitly requested
    if not config.getoption("--run-performance", default=False):
        skip_performance = pytest.mark.skip(reason="Performance tests skipped by default")
        for item in items:
            if "performance" in item.keywords:
                item.add_marker(skip_performance)


def pytest_addoption(parser):
    """Add command line options for performance tests."""
    parser.addoption(
        "--run-performance",
        action="store_true",
        default=False,
        help="Run performance tests"
    )
    parser.addoption(
        "--performance-baseline",
        action="store",
        default="performance_baseline.txt",
        help="Path to performance baseline file"
    )
    parser.addoption(
        "--performance-report",
        action="store",
        default="performance_report.txt",
        help="Path to performance report output"
    )