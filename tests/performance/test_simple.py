"""
Simple performance test to verify test structure works.
"""

import time
import pytest


@pytest.mark.performance
class TestSimplePerformance:
    """Simple performance tests that don't require external dependencies."""
    
    def test_basic_timing(self):
        """Test basic timing functionality."""
        start_time = time.time()
        
        # Simulate some work
        total = 0
        for i in range(10000):
            total += i
        
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Should complete quickly
        assert elapsed_time < 100  # Less than 100ms
        assert total == sum(range(10000))
        
        print(f"Basic computation took {elapsed_time:.2f}ms")
    
    @pytest.mark.stress
    def test_memory_allocation(self):
        """Test memory allocation performance."""
        import sys
        
        # Measure memory before
        initial_size = sys.getsizeof([])
        
        # Create large list
        large_list = list(range(100000))
        
        # Measure memory after
        final_size = sys.getsizeof(large_list)
        
        # Should allocate reasonable amount of memory
        memory_used = final_size - initial_size
        assert memory_used > 0
        assert len(large_list) == 100000
        
        print(f"Memory used for 100k items: {memory_used} bytes")
    
    @pytest.mark.regression
    def test_baseline_computation(self):
        """Test baseline computation performance."""
        iterations = 50000
        
        start_time = time.time()
        
        # Simple computation
        result = sum(i * i for i in range(iterations))
        
        elapsed_time = time.time() - start_time
        throughput = iterations / elapsed_time
        
        # Should maintain reasonable throughput
        assert throughput > 10000  # At least 10k operations per second
        assert result > 0
        
        print(f"Computation throughput: {throughput:.0f} ops/sec")
        
        # Log for regression tracking
        try:
            with open("performance_baseline.txt", "a") as f:
                f.write(f"COMPUTATION_BASELINE: {throughput:.0f} ops/sec\n")
        except Exception:
            pass  # Ignore file write errors in tests


@pytest.mark.performance
@pytest.mark.slow
class TestSlowPerformance:
    """Slower performance tests."""
    
    def test_sustained_computation(self):
        """Test sustained computation performance."""
        duration = 2.0  # 2 seconds
        start_time = time.time()
        
        operations = 0
        while (time.time() - start_time) < duration:
            # Simple operation
            _ = sum(range(1000))
            operations += 1
        
        actual_duration = time.time() - start_time
        ops_per_second = operations / actual_duration
        
        # Should maintain reasonable performance
        assert ops_per_second > 100  # At least 100 ops per second
        
        print(f"Sustained performance: {ops_per_second:.0f} ops/sec "
              f"over {actual_duration:.1f}s")


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"])