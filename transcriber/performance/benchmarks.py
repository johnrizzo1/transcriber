"""
Benchmarking suite for performance testing and regression detection.
"""

import asyncio
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
import json
import numpy as np

from .monitor import ComponentType, PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""
    name: str
    component: ComponentType
    timestamp: datetime
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    median_time_ms: float
    std_dev_ms: float
    percentile_95_ms: float
    percentile_99_ms: float
    success_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "name": self.name,
            "component": self.component.value,
            "timestamp": self.timestamp.isoformat(),
            "iterations": self.iterations,
            "total_time_ms": self.total_time_ms,
            "avg_time_ms": self.avg_time_ms,
            "min_time_ms": self.min_time_ms,
            "max_time_ms": self.max_time_ms,
            "median_time_ms": self.median_time_ms,
            "std_dev_ms": self.std_dev_ms,
            "percentile_95_ms": self.percentile_95_ms,
            "percentile_99_ms": self.percentile_99_ms,
            "success_rate": self.success_rate,
            "metadata": self.metadata
        }


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for the voice pipeline.
    
    Provides standardized performance tests for all components
    and end-to-end pipeline benchmarking.
    """
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Benchmark results storage
        self.results: List[BenchmarkResult] = []
        self.results_file = self.data_dir / "benchmark_results.json"
        
        # Load existing results
        self._load_results()
        
        # Performance monitor for detailed metrics
        self.performance_monitor: Optional[PerformanceMonitor] = None
    
    def set_performance_monitor(self, monitor: PerformanceMonitor) -> None:
        """Set performance monitor for detailed metrics collection."""
        self.performance_monitor = monitor
    
    def _load_results(self) -> None:
        """Load existing benchmark results."""
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    data = json.load(f)
                    self.results = [
                        BenchmarkResult(
                            name=r["name"],
                            component=ComponentType(r["component"]),
                            timestamp=datetime.fromisoformat(r["timestamp"]),
                            iterations=r["iterations"],
                            total_time_ms=r["total_time_ms"],
                            avg_time_ms=r["avg_time_ms"],
                            min_time_ms=r["min_time_ms"],
                            max_time_ms=r["max_time_ms"],
                            median_time_ms=r["median_time_ms"],
                            std_dev_ms=r["std_dev_ms"],
                            percentile_95_ms=r["percentile_95_ms"],
                            percentile_99_ms=r["percentile_99_ms"],
                            success_rate=r["success_rate"],
                            metadata=r.get("metadata", {})
                        )
                        for r in data.get("results", [])
                    ]
                logger.info(f"Loaded {len(self.results)} benchmark results")
            except Exception as e:
                logger.error(f"Error loading benchmark results: {e}")
                self.results = []
    
    def _save_results(self) -> None:
        """Save benchmark results to file."""
        try:
            data = {
                "last_updated": datetime.now().isoformat(),
                "total_results": len(self.results),
                "results": [r.to_dict() for r in self.results]
            }
            
            with open(self.results_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving benchmark results: {e}")
    
    async def run_benchmark(
        self,
        name: str,
        component: ComponentType,
        benchmark_func: Callable,
        iterations: int = 100,
        warmup_iterations: int = 10,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResult:
        """
        Run a benchmark test.
        
        Args:
            name: Benchmark name
            component: Component being tested
            benchmark_func: Function to benchmark (can be sync or async)
            iterations: Number of test iterations
            warmup_iterations: Number of warmup iterations
            metadata: Additional metadata
            
        Returns:
            Benchmark result
        """
        logger.info(f"Running benchmark: {name} ({iterations} iterations)")
        
        # Warmup phase
        logger.debug(f"Warmup phase: {warmup_iterations} iterations")
        for _ in range(warmup_iterations):
            try:
                if asyncio.iscoroutinefunction(benchmark_func):
                    await benchmark_func()
                else:
                    benchmark_func()
            except Exception as e:
                logger.warning(f"Warmup iteration failed: {e}")
        
        # Benchmark phase
        times = []
        successes = 0
        
        for i in range(iterations):
            start_time = time.perf_counter()
            success = True
            
            try:
                if asyncio.iscoroutinefunction(benchmark_func):
                    await benchmark_func()
                else:
                    benchmark_func()
                successes += 1
            except Exception as e:
                logger.debug(f"Benchmark iteration {i+1} failed: {e}")
                success = False
            
            end_time = time.perf_counter()
            elapsed_ms = (end_time - start_time) * 1000
            times.append(elapsed_ms)
            
            # Progress logging
            if (i + 1) % max(1, iterations // 10) == 0:
                logger.debug(f"Completed {i+1}/{iterations} iterations")
        
        # Calculate statistics
        total_time_ms = sum(times)
        avg_time_ms = statistics.mean(times)
        min_time_ms = min(times)
        max_time_ms = max(times)
        median_time_ms = statistics.median(times)
        std_dev_ms = statistics.stdev(times) if len(times) > 1 else 0.0
        percentile_95_ms = float(np.percentile(times, 95))
        percentile_99_ms = float(np.percentile(times, 99))
        success_rate = successes / iterations
        
        # Create result
        result = BenchmarkResult(
            name=name,
            component=component,
            timestamp=datetime.now(),
            iterations=iterations,
            total_time_ms=total_time_ms,
            avg_time_ms=avg_time_ms,
            min_time_ms=min_time_ms,
            max_time_ms=max_time_ms,
            median_time_ms=median_time_ms,
            std_dev_ms=std_dev_ms,
            percentile_95_ms=percentile_95_ms,
            percentile_99_ms=percentile_99_ms,
            success_rate=success_rate,
            metadata=metadata or {}
        )
        
        # Store result
        self.results.append(result)
        self._save_results()
        
        logger.info(f"Benchmark completed: {name} - "
                   f"avg: {avg_time_ms:.2f}ms, "
                   f"p95: {percentile_95_ms:.2f}ms, "
                   f"success: {success_rate:.1%}")
        
        return result
    
    async def benchmark_audio_capture(
        self, 
        audio_capture,
        duration_seconds: float = 1.0,
        iterations: int = 50
    ) -> BenchmarkResult:
        """Benchmark audio capture performance."""
        
        async def capture_test():
            """Test audio capture for specified duration."""
            chunks = []
            start_time = time.time()
            
            async with audio_capture:
                async for chunk in audio_capture.get_audio_chunks():
                    chunks.append(chunk)
                    if time.time() - start_time >= duration_seconds:
                        break
            
            return len(chunks)
        
        return await self.run_benchmark(
            name="audio_capture",
            component=ComponentType.AUDIO_CAPTURE,
            benchmark_func=capture_test,
            iterations=iterations,
            metadata={
                "duration_seconds": duration_seconds,
                "expected_chunks": int(duration_seconds * 10)  # Assuming 100ms chunks
            }
        )
    
    async def benchmark_vad_processing(
        self,
        vad_processor,
        test_audio_data: np.ndarray,
        iterations: int = 100
    ) -> BenchmarkResult:
        """Benchmark VAD processing performance."""
        
        def vad_test():
            """Test VAD processing on audio data."""
            return vad_processor.process_chunk(test_audio_data)
        
        return await self.run_benchmark(
            name="vad_processing",
            component=ComponentType.VAD,
            benchmark_func=vad_test,
            iterations=iterations,
            metadata={
                "audio_length_ms": len(test_audio_data) / 16,  # Assuming 16kHz
                "audio_samples": len(test_audio_data)
            }
        )
    
    async def benchmark_stt_processing(
        self,
        stt_processor,
        test_audio_file: str,
        iterations: int = 20
    ) -> BenchmarkResult:
        """Benchmark STT processing performance."""
        
        async def stt_test():
            """Test STT processing on audio file."""
            results = []
            async for result in stt_processor.transcribe_file(test_audio_file):
                results.append(result)
            return results
        
        return await self.run_benchmark(
            name="stt_processing",
            component=ComponentType.STT,
            benchmark_func=stt_test,
            iterations=iterations,
            metadata={
                "audio_file": test_audio_file,
                "file_size_mb": Path(test_audio_file).stat().st_size / (1024 * 1024)
            }
        )
    
    async def benchmark_llm_processing(
        self,
        llm_service,
        test_prompts: List[str],
        iterations: int = 30
    ) -> BenchmarkResult:
        """Benchmark LLM processing performance."""
        
        async def llm_test():
            """Test LLM processing with random prompt."""
            import random
            prompt = random.choice(test_prompts)
            response = await llm_service.process_user_input(prompt, "You are a helpful assistant.")
            return len(response)
        
        return await self.run_benchmark(
            name="llm_processing",
            component=ComponentType.LLM,
            benchmark_func=llm_test,
            iterations=iterations,
            metadata={
                "prompt_count": len(test_prompts),
                "avg_prompt_length": sum(len(p) for p in test_prompts) / len(test_prompts)
            }
        )
    
    async def benchmark_tts_processing(
        self,
        tts_service,
        test_texts: List[str],
        iterations: int = 50
    ) -> BenchmarkResult:
        """Benchmark TTS processing performance."""
        
        async def tts_test():
            """Test TTS processing with random text."""
            import random
            text = random.choice(test_texts)
            audio_data = await tts_service.speak(text)
            return len(audio_data)
        
        return await self.run_benchmark(
            name="tts_processing",
            component=ComponentType.TTS,
            benchmark_func=tts_test,
            iterations=iterations,
            metadata={
                "text_count": len(test_texts),
                "avg_text_length": sum(len(t) for t in test_texts) / len(test_texts)
            }
        )
    
    async def benchmark_end_to_end_latency(
        self,
        voice_pipeline,
        test_audio_file: str,
        iterations: int = 10
    ) -> BenchmarkResult:
        """Benchmark end-to-end pipeline latency."""
        
        async def e2e_test():
            """Test complete pipeline processing."""
            start_time = time.perf_counter()
            
            # Simulate pipeline processing
            # This would need to be adapted based on actual pipeline interface
            result = await voice_pipeline.process_audio_file(test_audio_file)
            
            end_time = time.perf_counter()
            return (end_time - start_time) * 1000
        
        return await self.run_benchmark(
            name="end_to_end_latency",
            component=ComponentType.PIPELINE,
            benchmark_func=e2e_test,
            iterations=iterations,
            metadata={
                "audio_file": test_audio_file
            }
        )
    
    async def run_stress_test(
        self,
        component: ComponentType,
        test_func: Callable,
        duration_minutes: int = 5,
        concurrent_requests: int = 10
    ) -> Dict[str, Any]:
        """
        Run stress test with concurrent requests.
        
        Args:
            component: Component being tested
            test_func: Function to stress test
            duration_minutes: Test duration in minutes
            concurrent_requests: Number of concurrent requests
            
        Returns:
            Stress test results
        """
        logger.info(f"Starting stress test: {component.value} "
                   f"({duration_minutes}min, {concurrent_requests} concurrent)")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        # Metrics tracking
        total_requests = 0
        successful_requests = 0
        failed_requests = 0
        response_times = []
        
        async def worker():
            """Worker coroutine for stress testing."""
            nonlocal total_requests, successful_requests, failed_requests
            
            while time.time() < end_time:
                request_start = time.perf_counter()
                total_requests += 1
                
                try:
                    if asyncio.iscoroutinefunction(test_func):
                        await test_func()
                    else:
                        test_func()
                    successful_requests += 1
                except Exception as e:
                    failed_requests += 1
                    logger.debug(f"Stress test request failed: {e}")
                
                request_end = time.perf_counter()
                response_times.append((request_end - request_start) * 1000)
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)
        
        # Run concurrent workers
        tasks = [asyncio.create_task(worker()) for _ in range(concurrent_requests)]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Stress test error: {e}")
        
        # Calculate results
        actual_duration = time.time() - start_time
        requests_per_second = total_requests / actual_duration
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        
        results = {
            "component": component.value,
            "duration_seconds": actual_duration,
            "concurrent_requests": concurrent_requests,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "requests_per_second": requests_per_second,
            "success_rate": success_rate,
            "avg_response_time_ms": statistics.mean(response_times) if response_times else 0,
            "p95_response_time_ms": np.percentile(response_times, 95) if response_times else 0,
            "p99_response_time_ms": np.percentile(response_times, 99) if response_times else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Stress test completed: {requests_per_second:.1f} req/s, "
                   f"{success_rate:.1%} success rate")
        
        return results
    
    def get_benchmark_history(
        self,
        name: Optional[str] = None,
        component: Optional[ComponentType] = None,
        days: int = 30
    ) -> List[BenchmarkResult]:
        """Get benchmark history with filtering."""
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        results = [
            r for r in self.results
            if r.timestamp >= cutoff_date
        ]
        
        if name:
            results = [r for r in results if r.name == name]
        
        if component:
            results = [r for r in results if r.component == component]
        
        return sorted(results, key=lambda x: x.timestamp, reverse=True)
    
    def detect_performance_regression(
        self,
        name: str,
        component: ComponentType,
        threshold_percent: float = 10.0
    ) -> Dict[str, Any]:
        """
        Detect performance regression by comparing recent results.
        
        Args:
            name: Benchmark name
            component: Component type
            threshold_percent: Regression threshold percentage
            
        Returns:
            Regression analysis results
        """
        # Get recent results for this benchmark
        recent_results = [
            r for r in self.results[-50:]  # Last 50 results
            if r.name == name and r.component == component
        ]
        
        if len(recent_results) < 5:
            return {
                "status": "insufficient_data",
                "message": f"Need at least 5 results, have {len(recent_results)}"
            }
        
        # Sort by timestamp
        recent_results.sort(key=lambda x: x.timestamp)
        
        # Compare latest result with baseline (average of previous results)
        latest = recent_results[-1]
        baseline_results = recent_results[-6:-1]  # Previous 5 results
        baseline_avg = statistics.mean([r.avg_time_ms for r in baseline_results])
        
        # Calculate regression
        regression_percent = ((latest.avg_time_ms - baseline_avg) / baseline_avg) * 100
        
        is_regression = regression_percent > threshold_percent
        
        return {
            "status": "regression_detected" if is_regression else "no_regression",
            "latest_avg_ms": latest.avg_time_ms,
            "baseline_avg_ms": baseline_avg,
            "regression_percent": regression_percent,
            "threshold_percent": threshold_percent,
            "is_regression": is_regression,
            "latest_timestamp": latest.timestamp.isoformat(),
            "baseline_count": len(baseline_results)
        }
    
    def export_benchmark_report(
        self,
        filepath: Union[str, Path],
        format: str = "json",
        days: int = 30
    ) -> None:
        """Export benchmark report to file."""
        filepath = Path(filepath)
        results = self.get_benchmark_history(days=days)
        
        if format.lower() == "json":
            data = {
                "export_timestamp": datetime.now().isoformat(),
                "days": days,
                "total_results": len(results),
                "results": [r.to_dict() for r in results]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format.lower() == "csv":
            import csv
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "name", "component", "timestamp", "iterations",
                    "avg_time_ms", "min_time_ms", "max_time_ms",
                    "median_time_ms", "p95_time_ms", "p99_time_ms",
                    "success_rate"
                ])
                
                for r in results:
                    writer.writerow([
                        r.name, r.component.value, r.timestamp.isoformat(),
                        r.iterations, r.avg_time_ms, r.min_time_ms,
                        r.max_time_ms, r.median_time_ms, r.percentile_95_ms,
                        r.percentile_99_ms, r.success_rate
                    ])
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported benchmark report to {filepath}")
    
    def clear_results(self, days: Optional[int] = None) -> int:
        """Clear benchmark results."""
        if days is None:
            # Clear all results
            count = len(self.results)
            self.results.clear()
        else:
            # Clear results older than specified days
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=days)
            
            old_results = [r for r in self.results if r.timestamp < cutoff_date]
            count = len(old_results)
            
            self.results = [r for r in self.results if r.timestamp >= cutoff_date]
        
        self._save_results()
        logger.info(f"Cleared {count} benchmark results")
        return count