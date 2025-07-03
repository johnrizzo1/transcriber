"""
Performance monitoring system for tracking pipeline metrics and resource usage.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import sqlite3
import threading
import psutil
import numpy as np

logger = logging.getLogger(__name__)


class ComponentType(Enum):
    """Types of pipeline components that can be monitored."""
    AUDIO_CAPTURE = "audio_capture"
    VAD = "vad"
    STT = "stt"
    AGENT = "agent"
    LLM = "llm"
    TOOL = "tool"
    TTS = "tts"
    AUDIO_OUTPUT = "audio_output"
    PIPELINE = "pipeline"


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    component: ComponentType
    timestamp: datetime
    latency_ms: float
    memory_mb: float
    cpu_percent: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "component": self.component.value,
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": self.latency_ms,
            "memory_mb": self.memory_mb,
            "cpu_percent": self.cpu_percent,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceMetrics":
        """Create metrics from dictionary."""
        return cls(
            component=ComponentType(data["component"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            latency_ms=data["latency_ms"],
            memory_mb=data["memory_mb"],
            cpu_percent=data["cpu_percent"],
            metadata=data.get("metadata", {})
        )


@dataclass
class ResourceSnapshot:
    """Snapshot of system resource usage."""
    timestamp: datetime
    memory_total_mb: float
    memory_used_mb: float
    memory_percent: float
    cpu_percent: float
    cpu_count: int
    disk_usage_percent: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "memory_total_mb": self.memory_total_mb,
            "memory_used_mb": self.memory_used_mb,
            "memory_percent": self.memory_percent,
            "cpu_percent": self.cpu_percent,
            "cpu_count": self.cpu_count,
            "disk_usage_percent": self.disk_usage_percent
        }


class PerformanceMonitor:
    """
    Main performance monitoring system.
    
    Tracks performance metrics for all pipeline components and provides
    real-time monitoring, historical analysis, and optimization insights.
    """
    
    def __init__(self, data_dir: str = "./data", enable_storage: bool = True):
        self.data_dir = Path(data_dir)
        self.enable_storage = enable_storage
        
        # In-memory metrics storage
        self.metrics: List[PerformanceMetrics] = []
        self.resource_snapshots: List[ResourceSnapshot] = []
        
        # Performance tracking
        self._active_timers: Dict[str, float] = {}
        self._component_stats: Dict[ComponentType, Dict[str, Any]] = {}
        
        # Storage
        self.db_path = self.data_dir / "performance.db"
        self._db_lock = threading.Lock()
        
        # Monitoring state
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._monitor_interval = 1.0  # seconds
        
        # Process monitoring
        self._process = psutil.Process()
        
        # Initialize storage
        if self.enable_storage:
            self._init_storage()
    
    def _init_storage(self) -> None:
        """Initialize SQLite storage for performance data."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    memory_mb REAL NOT NULL,
                    cpu_percent REAL NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS resource_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    memory_total_mb REAL NOT NULL,
                    memory_used_mb REAL NOT NULL,
                    memory_percent REAL NOT NULL,
                    cpu_percent REAL NOT NULL,
                    cpu_count INTEGER NOT NULL,
                    disk_usage_percent REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better query performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_component_timestamp 
                ON performance_metrics(component, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp 
                ON resource_snapshots(timestamp)
            """)
            
            conn.commit()
    
    async def start_monitoring(self, interval: float = 1.0) -> None:
        """Start continuous performance monitoring."""
        if self._monitoring:
            logger.warning("Performance monitoring already running")
            return
        
        self._monitor_interval = interval
        self._monitoring = True
        
        logger.info(f"Starting performance monitoring (interval: {interval}s)")
        
        # Start background monitoring task
        self._monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        if not self._monitoring:
            return
        
        logger.info("Stopping performance monitoring")
        self._monitoring = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
    
    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        try:
            while self._monitoring:
                # Take resource snapshot
                snapshot = self._take_resource_snapshot()
                self.resource_snapshots.append(snapshot)
                
                # Store to database if enabled
                if self.enable_storage:
                    await self._store_resource_snapshot(snapshot)
                
                # Clean up old in-memory data (keep last 1000 entries)
                if len(self.resource_snapshots) > 1000:
                    self.resource_snapshots = self.resource_snapshots[-1000:]
                
                await asyncio.sleep(self._monitor_interval)
                
        except asyncio.CancelledError:
            logger.debug("Performance monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in performance monitoring loop: {e}")
    
    def _take_resource_snapshot(self) -> ResourceSnapshot:
        """Take a snapshot of current system resources."""
        try:
            # Get memory info
            memory = psutil.virtual_memory()
            
            # Get CPU info
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_count = psutil.cpu_count() or 1  # Default to 1 if None
            
            # Get disk usage for data directory
            disk_usage = psutil.disk_usage(str(self.data_dir.parent))
            
            return ResourceSnapshot(
                timestamp=datetime.now(),
                memory_total_mb=memory.total / (1024 * 1024),
                memory_used_mb=memory.used / (1024 * 1024),
                memory_percent=memory.percent,
                cpu_percent=cpu_percent,
                cpu_count=cpu_count,
                disk_usage_percent=disk_usage.percent
            )
        except Exception as e:
            logger.error(f"Error taking resource snapshot: {e}")
            # Return default snapshot
            return ResourceSnapshot(
                timestamp=datetime.now(),
                memory_total_mb=0.0,
                memory_used_mb=0.0,
                memory_percent=0.0,
                cpu_percent=0.0,
                cpu_count=1,
                disk_usage_percent=0.0
            )
    
    def start_timer(self, component: ComponentType, operation: str = "default") -> str:
        """Start timing an operation."""
        timer_key = f"{component.value}:{operation}"
        self._active_timers[timer_key] = time.perf_counter()
        return timer_key
    
    def end_timer(self, timer_key: str, metadata: Optional[Dict[str, Any]] = None) -> float:
        """End timing and record metrics."""
        if timer_key not in self._active_timers:
            logger.warning(f"Timer not found: {timer_key}")
            return 0.0
        
        # Calculate elapsed time
        start_time = self._active_timers.pop(timer_key)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Parse component from timer key
        component_str = timer_key.split(":")[0]
        try:
            component = ComponentType(component_str)
        except ValueError:
            logger.error(f"Invalid component in timer key: {component_str}")
            return elapsed_ms
        
        # Get current resource usage
        try:
            memory_mb = self._process.memory_info().rss / (1024 * 1024)
            cpu_percent = self._process.cpu_percent()
        except Exception as e:
            logger.warning(f"Error getting process metrics: {e}")
            memory_mb = 0.0
            cpu_percent = 0.0
        
        # Create metrics
        metrics = PerformanceMetrics(
            component=component,
            timestamp=datetime.now(),
            latency_ms=elapsed_ms,
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            metadata=metadata or {}
        )
        
        # Store metrics
        self.record_metrics(metrics)
        
        return elapsed_ms
    
    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics."""
        # Add to in-memory storage
        self.metrics.append(metrics)
        
        # Update component statistics
        self._update_component_stats(metrics)
        
        # Store to database if enabled
        if self.enable_storage:
            asyncio.create_task(self._store_metrics(metrics))
        
        # Clean up old in-memory data (keep last 10000 entries)
        if len(self.metrics) > 10000:
            self.metrics = self.metrics[-10000:]
    
    def _update_component_stats(self, metrics: PerformanceMetrics) -> None:
        """Update running statistics for a component."""
        component = metrics.component
        
        if component not in self._component_stats:
            self._component_stats[component] = {
                "count": 0,
                "total_latency": 0.0,
                "min_latency": float("inf"),
                "max_latency": 0.0,
                "avg_memory": 0.0,
                "avg_cpu": 0.0
            }
        
        stats = self._component_stats[component]
        stats["count"] += 1
        stats["total_latency"] += metrics.latency_ms
        stats["min_latency"] = min(stats["min_latency"], metrics.latency_ms)
        stats["max_latency"] = max(stats["max_latency"], metrics.latency_ms)
        
        # Update running averages
        alpha = 0.1  # Exponential moving average factor
        stats["avg_memory"] = (1 - alpha) * stats["avg_memory"] + alpha * metrics.memory_mb
        stats["avg_cpu"] = (1 - alpha) * stats["avg_cpu"] + alpha * metrics.cpu_percent
    
    async def _store_metrics(self, metrics: PerformanceMetrics) -> None:
        """Store metrics to database."""
        if not self.enable_storage:
            return
        
        try:
            with self._db_lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO performance_metrics 
                        (component, timestamp, latency_ms, memory_mb, cpu_percent, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        metrics.component.value,
                        metrics.timestamp.isoformat(),
                        metrics.latency_ms,
                        metrics.memory_mb,
                        metrics.cpu_percent,
                        json.dumps(metrics.metadata)
                    ))
                    conn.commit()
        except Exception as e:
            logger.error(f"Error storing metrics to database: {e}")
    
    async def _store_resource_snapshot(self, snapshot: ResourceSnapshot) -> None:
        """Store resource snapshot to database."""
        if not self.enable_storage:
            return
        
        try:
            with self._db_lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO resource_snapshots 
                        (timestamp, memory_total_mb, memory_used_mb, memory_percent,
                         cpu_percent, cpu_count, disk_usage_percent)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        snapshot.timestamp.isoformat(),
                        snapshot.memory_total_mb,
                        snapshot.memory_used_mb,
                        snapshot.memory_percent,
                        snapshot.cpu_percent,
                        snapshot.cpu_count,
                        snapshot.disk_usage_percent
                    ))
                    conn.commit()
        except Exception as e:
            logger.error(f"Error storing resource snapshot to database: {e}")
    
    def get_component_stats(self, component: ComponentType) -> Dict[str, Any]:
        """Get statistics for a specific component."""
        if component not in self._component_stats:
            return {}
        
        stats = self._component_stats[component].copy()
        if stats["count"] > 0:
            stats["avg_latency"] = stats["total_latency"] / stats["count"]
        else:
            stats["avg_latency"] = 0.0
        
        return stats
    
    def get_recent_metrics(
        self, 
        component: Optional[ComponentType] = None,
        minutes: int = 10
    ) -> List[PerformanceMetrics]:
        """Get recent metrics for analysis."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        recent_metrics = [
            m for m in self.metrics 
            if m.timestamp >= cutoff_time
        ]
        
        if component:
            recent_metrics = [
                m for m in recent_metrics 
                if m.component == component
            ]
        
        return recent_metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        summary = {
            "total_metrics": len(self.metrics),
            "monitoring_active": self._monitoring,
            "components": {}
        }
        
        for component, stats in self._component_stats.items():
            component_summary = stats.copy()
            if stats["count"] > 0:
                component_summary["avg_latency"] = stats["total_latency"] / stats["count"]
            else:
                component_summary["avg_latency"] = 0.0
            
            summary["components"][component.value] = component_summary
        
        # Add recent resource usage
        if self.resource_snapshots:
            latest_snapshot = self.resource_snapshots[-1]
            summary["current_resources"] = latest_snapshot.to_dict()
        
        return summary
    
    async def export_metrics(
        self, 
        filepath: Union[str, Path],
        format: str = "json",
        component: Optional[ComponentType] = None,
        hours: int = 24
    ) -> None:
        """Export metrics to file."""
        filepath = Path(filepath)
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter metrics
        export_metrics = [
            m for m in self.metrics 
            if m.timestamp >= cutoff_time
        ]
        
        if component:
            export_metrics = [
                m for m in export_metrics 
                if m.component == component
            ]
        
        # Export based on format
        if format.lower() == "json":
            data = {
                "export_timestamp": datetime.now().isoformat(),
                "component_filter": component.value if component else None,
                "hours": hours,
                "metrics": [m.to_dict() for m in export_metrics]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format.lower() == "csv":
            import csv
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "component", "timestamp", "latency_ms", 
                    "memory_mb", "cpu_percent", "metadata"
                ])
                
                for m in export_metrics:
                    writer.writerow([
                        m.component.value,
                        m.timestamp.isoformat(),
                        m.latency_ms,
                        m.memory_mb,
                        m.cpu_percent,
                        json.dumps(m.metadata)
                    ])
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported {len(export_metrics)} metrics to {filepath}")
    
    async def cleanup_old_data(self, days: int = 7) -> int:
        """Clean up old performance data."""
        if not self.enable_storage:
            return 0
        
        cutoff_time = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_time.isoformat()
        
        deleted_count = 0
        
        try:
            with self._db_lock:
                with sqlite3.connect(self.db_path) as conn:
                    # Delete old metrics
                    cursor = conn.execute(
                        "DELETE FROM performance_metrics WHERE timestamp < ?",
                        (cutoff_str,)
                    )
                    deleted_count += cursor.rowcount
                    
                    # Delete old snapshots
                    cursor = conn.execute(
                        "DELETE FROM resource_snapshots WHERE timestamp < ?",
                        (cutoff_str,)
                    )
                    deleted_count += cursor.rowcount
                    
                    conn.commit()
                    
                    # Vacuum database to reclaim space
                    conn.execute("VACUUM")
            
            logger.info(f"Cleaned up {deleted_count} old performance records")
            
        except Exception as e:
            logger.error(f"Error cleaning up old performance data: {e}")
        
        return deleted_count


# Context manager for easy performance timing
class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(
        self, 
        monitor: PerformanceMonitor,
        component: ComponentType,
        operation: str = "default",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.monitor = monitor
        self.component = component
        self.operation = operation
        self.metadata = metadata or {}
        self.timer_key: Optional[str] = None
    
    def __enter__(self) -> "PerformanceTimer":
        self.timer_key = self.monitor.start_timer(self.component, self.operation)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.timer_key:
            elapsed_ms = self.monitor.end_timer(self.timer_key, self.metadata)
            logger.debug(f"{self.component.value}:{self.operation} took {elapsed_ms:.2f}ms")


# Decorator for automatic performance monitoring
def monitor_performance(
    component: ComponentType,
    operation: str = "default",
    monitor_instance: Optional[PerformanceMonitor] = None
):
    """Decorator to automatically monitor function performance."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            nonlocal monitor_instance
            if monitor_instance is None:
                # Try to get monitor from first argument if it has one
                if args and hasattr(args[0], 'performance_monitor'):
                    monitor_instance = args[0].performance_monitor
                else:
                    # Skip monitoring if no monitor available
                    return await func(*args, **kwargs)
            
            # Type check to ensure we have a valid monitor
            if not isinstance(monitor_instance, PerformanceMonitor):
                return await func(*args, **kwargs)
            
            with PerformanceTimer(monitor_instance, component, operation):
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            nonlocal monitor_instance
            if monitor_instance is None:
                # Try to get monitor from first argument if it has one
                if args and hasattr(args[0], 'performance_monitor'):
                    monitor_instance = args[0].performance_monitor
                else:
                    # Skip monitoring if no monitor available
                    return func(*args, **kwargs)
            
            # Type check to ensure we have a valid monitor
            if not isinstance(monitor_instance, PerformanceMonitor):
                return func(*args, **kwargs)
            
            with PerformanceTimer(monitor_instance, component, operation):
                return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator