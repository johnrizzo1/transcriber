"""
Profiling system for identifying performance bottlenecks in the voice pipeline.
"""

import asyncio
import cProfile
import io
import logging
import pstats
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import functools
import threading
import traceback

from .monitor import ComponentType, PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Result of a profiling session."""
    component: ComponentType
    operation: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    call_count: int
    total_time: float
    cumulative_time: float
    top_functions: List[Dict[str, Any]] = field(default_factory=list)
    memory_profile: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "component": self.component.value,
            "operation": self.operation,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_ms": self.duration_ms,
            "call_count": self.call_count,
            "total_time": self.total_time,
            "cumulative_time": self.cumulative_time,
            "top_functions": self.top_functions,
            "memory_profile": self.memory_profile,
            "metadata": self.metadata
        }


class ProfilerManager:
    """
    Manager for profiling pipeline components and operations.
    
    Provides both statistical profiling (cProfile) and custom timing
    profiling for detailed performance analysis.
    """
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Active profiling sessions
        self._active_profilers: Dict[str, cProfile.Profile] = {}
        self._profile_results: List[ProfileResult] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Memory profiling (optional)
        self._memory_profiling_enabled = False
        try:
            import tracemalloc
            self._tracemalloc = tracemalloc
            self._memory_profiling_enabled = True
        except ImportError:
            logger.warning("tracemalloc not available, memory profiling disabled")
            self._tracemalloc = None
    
    def start_profiling(
        self, 
        component: ComponentType, 
        operation: str = "default",
        enable_memory: bool = False
    ) -> str:
        """Start profiling a component operation."""
        profile_key = f"{component.value}:{operation}:{int(time.time())}"
        
        with self._lock:
            # Create new profiler
            profiler = cProfile.Profile()
            profiler.enable()
            
            self._active_profilers[profile_key] = profiler
            
            # Start memory profiling if requested and available
            if enable_memory and self._memory_profiling_enabled and self._tracemalloc:
                if not self._tracemalloc.is_tracing():
                    self._tracemalloc.start()
        
        logger.debug(f"Started profiling: {profile_key}")
        return profile_key
    
    def stop_profiling(
        self, 
        profile_key: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProfileResult:
        """Stop profiling and return results."""
        with self._lock:
            if profile_key not in self._active_profilers:
                raise ValueError(f"Profile session not found: {profile_key}")
            
            # Stop profiler
            profiler = self._active_profilers.pop(profile_key)
            profiler.disable()
            
            # Parse profile key
            parts = profile_key.split(":")
            component = ComponentType(parts[0])
            operation = parts[1]
            start_timestamp = int(parts[2])
            
            # Analyze profiling results
            result = self._analyze_profile(
                profiler, component, operation, start_timestamp, metadata
            )
            
            # Store result
            self._profile_results.append(result)
            
            # Clean up old results (keep last 100)
            if len(self._profile_results) > 100:
                self._profile_results = self._profile_results[-100:]
            
            logger.debug(f"Stopped profiling: {profile_key}")
            return result
    
    def _analyze_profile(
        self,
        profiler: cProfile.Profile,
        component: ComponentType,
        operation: str,
        start_timestamp: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProfileResult:
        """Analyze profiling results."""
        end_time = datetime.now()
        start_time = datetime.fromtimestamp(start_timestamp)
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Get profiling statistics
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats('cumulative')
        
        # Extract top functions
        top_functions = []
        try:
            # Get function statistics - access the internal stats dict
            stats_dict = getattr(stats, 'stats', {})
            for func_key, (call_count, total_time, cumulative_time, callers) in stats_dict.items():
                filename, line_number, function_name = func_key
                
                # Skip internal Python functions
                if filename.startswith('<') or 'site-packages' in filename:
                    continue
                
                top_functions.append({
                    "function": function_name,
                    "filename": filename,
                    "line_number": line_number,
                    "call_count": call_count,
                    "total_time": total_time,
                    "cumulative_time": cumulative_time,
                    "time_per_call": total_time / call_count if call_count > 0 else 0
                })
            
            # Sort by cumulative time and take top 20
            top_functions.sort(key=lambda x: x["cumulative_time"], reverse=True)
            top_functions = top_functions[:20]
            
        except Exception as e:
            logger.error(f"Error analyzing profile statistics: {e}")
            top_functions = []
        
        # Get memory profile if available
        memory_profile = None
        if self._memory_profiling_enabled and self._tracemalloc and self._tracemalloc.is_tracing():
            try:
                current, peak = self._tracemalloc.get_traced_memory()
                memory_profile = {
                    "current_mb": current / (1024 * 1024),
                    "peak_mb": peak / (1024 * 1024)
                }
            except Exception as e:
                logger.error(f"Error getting memory profile: {e}")
        
        # Calculate totals
        stats_dict = getattr(stats, 'stats', {})
        total_call_count = 0
        total_time = 0.0
        total_cumulative = 0.0
        
        # Handle different cProfile stats formats
        for stat_values in stats_dict.values():
            if len(stat_values) >= 4:
                # Format: (call_count, recursive_count, total_time, cumulative_time, ...)
                call_count, _, tt, ct = stat_values[:4]
                total_call_count += call_count
                total_time += tt
                total_cumulative += ct
            elif len(stat_values) >= 2:
                # Fallback format
                call_count, tt = stat_values[:2]
                total_call_count += call_count
                total_time += tt
                total_cumulative += tt
        
        return ProfileResult(
            component=component,
            operation=operation,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            call_count=total_call_count,
            total_time=total_time,
            cumulative_time=total_cumulative,
            top_functions=top_functions,
            memory_profile=memory_profile,
            metadata=metadata or {}
        )
    
    def get_profile_results(
        self, 
        component: Optional[ComponentType] = None,
        operation: Optional[str] = None,
        limit: int = 10
    ) -> List[ProfileResult]:
        """Get stored profile results."""
        results = self._profile_results.copy()
        
        # Filter by component
        if component:
            results = [r for r in results if r.component == component]
        
        # Filter by operation
        if operation:
            results = [r for r in results if r.operation == operation]
        
        # Sort by end time (most recent first) and limit
        results.sort(key=lambda x: x.end_time, reverse=True)
        return results[:limit]
    
    def export_profile_report(
        self, 
        filepath: Union[str, Path],
        component: Optional[ComponentType] = None,
        format: str = "text"
    ) -> None:
        """Export profiling report to file."""
        filepath = Path(filepath)
        results = self.get_profile_results(component=component, limit=50)
        
        if format.lower() == "text":
            self._export_text_report(filepath, results)
        elif format.lower() == "json":
            self._export_json_report(filepath, results)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported profile report to {filepath}")
    
    def _export_text_report(self, filepath: Path, results: List[ProfileResult]) -> None:
        """Export text format report."""
        with open(filepath, 'w') as f:
            f.write("PERFORMANCE PROFILING REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            if not results:
                f.write("No profiling results available.\n")
                return
            
            # Summary statistics
            f.write("SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total profiles: {len(results)}\n")
            
            # Group by component
            component_stats = {}
            for result in results:
                comp = result.component.value
                if comp not in component_stats:
                    component_stats[comp] = {
                        "count": 0,
                        "total_duration": 0.0,
                        "avg_duration": 0.0
                    }
                component_stats[comp]["count"] += 1
                component_stats[comp]["total_duration"] += result.duration_ms
            
            for comp, stats in component_stats.items():
                stats["avg_duration"] = stats["total_duration"] / stats["count"]
                f.write(f"{comp}: {stats['count']} profiles, "
                       f"avg {stats['avg_duration']:.2f}ms\n")
            
            f.write("\n")
            
            # Detailed results
            f.write("DETAILED RESULTS\n")
            f.write("-" * 20 + "\n\n")
            
            for i, result in enumerate(results[:10], 1):
                f.write(f"{i}. {result.component.value}:{result.operation}\n")
                f.write(f"   Duration: {result.duration_ms:.2f}ms\n")
                f.write(f"   Calls: {result.call_count}\n")
                f.write(f"   Time: {result.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                if result.memory_profile:
                    f.write(f"   Memory: {result.memory_profile['peak_mb']:.2f}MB peak\n")
                
                # Top functions
                if result.top_functions:
                    f.write("   Top functions:\n")
                    for func in result.top_functions[:5]:
                        f.write(f"     - {func['function']}: "
                               f"{func['cumulative_time']:.4f}s "
                               f"({func['call_count']} calls)\n")
                
                f.write("\n")
    
    def _export_json_report(self, filepath: Path, results: List[ProfileResult]) -> None:
        """Export JSON format report."""
        import json
        
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_results": len(results),
            "results": [result.to_dict() for result in results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def clear_results(self) -> None:
        """Clear stored profiling results."""
        with self._lock:
            self._profile_results.clear()
        logger.info("Cleared profiling results")
    
    def get_bottleneck_analysis(
        self, 
        component: Optional[ComponentType] = None
    ) -> Dict[str, Any]:
        """Analyze bottlenecks across profiling results."""
        results = self.get_profile_results(component=component, limit=50)
        
        if not results:
            return {"error": "No profiling results available"}
        
        # Aggregate function statistics
        function_stats = {}
        total_profiles = len(results)
        
        for result in results:
            for func in result.top_functions:
                func_key = f"{func['function']}:{func['filename']}"
                
                if func_key not in function_stats:
                    function_stats[func_key] = {
                        "function": func['function'],
                        "filename": func['filename'],
                        "total_time": 0.0,
                        "total_calls": 0,
                        "appearances": 0,
                        "avg_time_per_call": 0.0
                    }
                
                stats = function_stats[func_key]
                stats["total_time"] += func["cumulative_time"]
                stats["total_calls"] += func["call_count"]
                stats["appearances"] += 1
        
        # Calculate averages and sort by impact
        for stats in function_stats.values():
            if stats["total_calls"] > 0:
                stats["avg_time_per_call"] = stats["total_time"] / stats["total_calls"]
            stats["frequency"] = stats["appearances"] / total_profiles
        
        # Sort by total time (biggest bottlenecks first)
        bottlenecks = sorted(
            function_stats.values(),
            key=lambda x: x["total_time"],
            reverse=True
        )[:10]
        
        # Component-level analysis
        component_analysis = {}
        for result in results:
            comp = result.component.value
            if comp not in component_analysis:
                component_analysis[comp] = {
                    "count": 0,
                    "total_duration": 0.0,
                    "avg_duration": 0.0,
                    "min_duration": float("inf"),
                    "max_duration": 0.0
                }
            
            stats = component_analysis[comp]
            stats["count"] += 1
            stats["total_duration"] += result.duration_ms
            stats["min_duration"] = min(stats["min_duration"], result.duration_ms)
            stats["max_duration"] = max(stats["max_duration"], result.duration_ms)
        
        for stats in component_analysis.values():
            stats["avg_duration"] = stats["total_duration"] / stats["count"]
        
        return {
            "total_profiles_analyzed": total_profiles,
            "top_bottlenecks": bottlenecks,
            "component_analysis": component_analysis,
            "analysis_timestamp": datetime.now().isoformat()
        }


# Context manager for easy profiling
class ProfilerContext:
    """Context manager for profiling operations."""
    
    def __init__(
        self,
        profiler_manager: ProfilerManager,
        component: ComponentType,
        operation: str = "default",
        enable_memory: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.profiler_manager = profiler_manager
        self.component = component
        self.operation = operation
        self.enable_memory = enable_memory
        self.metadata = metadata or {}
        self.profile_key: Optional[str] = None
        self.result: Optional[ProfileResult] = None
    
    def __enter__(self) -> "ProfilerContext":
        self.profile_key = self.profiler_manager.start_profiling(
            self.component, self.operation, self.enable_memory
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.profile_key:
            self.result = self.profiler_manager.stop_profiling(
                self.profile_key, self.metadata
            )
            logger.debug(f"Profiled {self.component.value}:{self.operation} - "
                        f"{self.result.duration_ms:.2f}ms")


# Decorator for automatic profiling
def profile_performance(
    component: ComponentType,
    operation: str = "default",
    enable_memory: bool = False,
    profiler_manager: Optional[ProfilerManager] = None
):
    """Decorator to automatically profile function performance."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            nonlocal profiler_manager
            if profiler_manager is None:
                # Try to get profiler from first argument if it has one
                if args and hasattr(args[0], 'profiler_manager'):
                    profiler_manager = args[0].profiler_manager
                else:
                    # Skip profiling if no profiler available
                    return await func(*args, **kwargs)
            
            # Type check to ensure we have a valid profiler
            if not isinstance(profiler_manager, ProfilerManager):
                return await func(*args, **kwargs)
            
            with ProfilerContext(profiler_manager, component, operation, enable_memory):
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            nonlocal profiler_manager
            if profiler_manager is None:
                # Try to get profiler from first argument if it has one
                if args and hasattr(args[0], 'profiler_manager'):
                    profiler_manager = args[0].profiler_manager
                else:
                    # Skip profiling if no profiler available
                    return func(*args, **kwargs)
            
            # Type check to ensure we have a valid profiler
            if not isinstance(profiler_manager, ProfilerManager):
                return func(*args, **kwargs)
            
            with ProfilerContext(profiler_manager, component, operation, enable_memory):
                return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator