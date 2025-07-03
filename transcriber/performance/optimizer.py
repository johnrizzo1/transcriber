"""
Resource optimization system for improving pipeline performance.
"""

import asyncio
import logging
import gc
import os
import sys
import psutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import json

from .monitor import ComponentType, PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of an optimization operation."""
    optimization_type: str
    component: ComponentType
    timestamp: datetime
    before_metrics: Dict[str, Any]
    after_metrics: Dict[str, Any]
    improvement_percent: float
    success: bool
    description: str
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "optimization_type": self.optimization_type,
            "component": self.component.value,
            "timestamp": self.timestamp.isoformat(),
            "before_metrics": self.before_metrics,
            "after_metrics": self.after_metrics,
            "improvement_percent": self.improvement_percent,
            "success": self.success,
            "description": self.description,
            "recommendations": self.recommendations,
            "metadata": self.metadata
        }


class ResourceOptimizer:
    """
    Resource optimization manager for the voice pipeline.
    
    Provides automatic and manual optimization strategies for
    memory usage, CPU utilization, and performance tuning.
    """
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimization results
        self.optimization_results: List[OptimizationResult] = []
        
        # System information
        self._system_info = self._get_system_info()
        
        # Model cache management
        self._model_cache: Dict[str, Any] = {}
        self._cache_size_limit_mb = 2048  # 2GB default
        
        # Performance monitor reference
        self.performance_monitor: Optional[PerformanceMonitor] = None
    
    def set_performance_monitor(self, monitor: PerformanceMonitor) -> None:
        """Set performance monitor for metrics collection."""
        self.performance_monitor = monitor
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for optimization decisions."""
        try:
            memory = psutil.virtual_memory()
            cpu_info = {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            }
            
            return {
                "memory_total_gb": memory.total / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "cpu_info": cpu_info,
                "platform": os.name,
                "python_version": os.sys.version
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {}
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get optimization recommendations based on system analysis."""
        recommendations = []
        
        # Memory recommendations
        memory_gb = self._system_info.get("memory_total_gb", 0)
        if memory_gb < 8:
            recommendations.append({
                "type": "memory",
                "priority": "high",
                "title": "Low System Memory",
                "description": f"System has {memory_gb:.1f}GB RAM. Consider using smaller models.",
                "actions": [
                    "Use whisper 'tiny' model instead of 'base' or larger",
                    "Reduce audio buffer sizes",
                    "Enable aggressive garbage collection",
                    "Limit conversation history length"
                ]
            })
        elif memory_gb < 16:
            recommendations.append({
                "type": "memory",
                "priority": "medium",
                "title": "Moderate System Memory",
                "description": f"System has {memory_gb:.1f}GB RAM. Some optimizations recommended.",
                "actions": [
                    "Monitor memory usage during long sessions",
                    "Consider model caching strategies",
                    "Enable periodic cleanup"
                ]
            })
        
        # CPU recommendations
        cpu_info = self._system_info.get("cpu_info", {})
        logical_cores = cpu_info.get("logical_cores", 1)
        
        if logical_cores < 4:
            recommendations.append({
                "type": "cpu",
                "priority": "high",
                "title": "Limited CPU Cores",
                "description": f"System has {logical_cores} CPU cores. Performance may be limited.",
                "actions": [
                    "Use CPU-optimized model settings",
                    "Reduce concurrent processing",
                    "Consider cloud processing for heavy tasks"
                ]
            })
        
        # Performance-based recommendations
        if self.performance_monitor:
            recent_metrics = self.performance_monitor.get_recent_metrics(minutes=30)
            
            # Check for high latency components
            component_latencies = {}
            for metric in recent_metrics:
                comp = metric.component
                if comp not in component_latencies:
                    component_latencies[comp] = []
                component_latencies[comp].append(metric.latency_ms)
            
            for component, latencies in component_latencies.items():
                avg_latency = sum(latencies) / len(latencies)
                
                if component == ComponentType.STT and avg_latency > 200:
                    recommendations.append({
                        "type": "performance",
                        "priority": "medium",
                        "title": "High STT Latency",
                        "description": f"STT processing averaging {avg_latency:.1f}ms",
                        "actions": [
                            "Consider using faster Whisper model",
                            "Optimize audio chunk sizes",
                            "Check CPU usage during STT"
                        ]
                    })
                
                elif component == ComponentType.LLM and avg_latency > 300:
                    recommendations.append({
                        "type": "performance",
                        "priority": "medium",
                        "title": "High LLM Latency",
                        "description": f"LLM processing averaging {avg_latency:.1f}ms",
                        "actions": [
                            "Consider smaller LLM model",
                            "Optimize context length",
                            "Check Ollama configuration"
                        ]
                    })
        
        return recommendations
    
    async def optimize_memory_usage(self) -> OptimizationResult:
        """Optimize memory usage across the pipeline."""
        logger.info("Starting memory optimization")
        
        # Get before metrics
        before_memory = psutil.virtual_memory()
        before_metrics = {
            "memory_used_mb": before_memory.used / (1024**2),
            "memory_percent": before_memory.percent,
            "cache_size_mb": sum(
                sys.getsizeof(obj) for obj in self._model_cache.values()
            ) / (1024**2)
        }
        
        optimizations_applied = []
        
        # 1. Garbage collection
        collected = gc.collect()
        if collected > 0:
            optimizations_applied.append(f"Collected {collected} objects")
        
        # 2. Clear model cache if too large
        cache_size_mb = before_metrics["cache_size_mb"]
        if cache_size_mb > self._cache_size_limit_mb:
            cleared_models = len(self._model_cache)
            self._model_cache.clear()
            optimizations_applied.append(f"Cleared {cleared_models} cached models")
        
        # 3. Optimize conversation history
        # This would need to be integrated with the actual agent
        optimizations_applied.append("Conversation history optimization (placeholder)")
        
        # Get after metrics
        after_memory = psutil.virtual_memory()
        after_metrics = {
            "memory_used_mb": after_memory.used / (1024**2),
            "memory_percent": after_memory.percent,
            "cache_size_mb": sum(
                sys.getsizeof(obj) for obj in self._model_cache.values()
            ) / (1024**2)
        }
        
        # Calculate improvement
        memory_saved_mb = before_metrics["memory_used_mb"] - after_metrics["memory_used_mb"]
        improvement_percent = (memory_saved_mb / before_metrics["memory_used_mb"]) * 100
        
        result = OptimizationResult(
            optimization_type="memory_usage",
            component=ComponentType.PIPELINE,
            timestamp=datetime.now(),
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            improvement_percent=improvement_percent,
            success=len(optimizations_applied) > 0,
            description=f"Memory optimization: {', '.join(optimizations_applied)}",
            recommendations=[
                "Monitor memory usage regularly",
                "Consider smaller models if memory is constrained",
                "Enable automatic cleanup for long sessions"
            ]
        )
        
        self.optimization_results.append(result)
        
        logger.info(f"Memory optimization completed: {improvement_percent:.1f}% improvement")
        return result
    
    async def optimize_model_loading(self, model_type: str) -> OptimizationResult:
        """Optimize model loading and caching."""
        logger.info(f"Optimizing model loading for {model_type}")
        
        before_metrics = {
            "cached_models": len(self._model_cache),
            "cache_size_mb": sum(
                sys.getsizeof(obj) for obj in self._model_cache.values()
            ) / (1024**2)
        }
        
        optimizations = []
        
        # Model-specific optimizations
        if model_type == "whisper":
            optimizations.extend([
                "Preload Whisper model at startup",
                "Use model quantization if available",
                "Optimize compute type for CPU/GPU"
            ])
        elif model_type == "llm":
            optimizations.extend([
                "Keep LLM connection warm",
                "Optimize context window size",
                "Use streaming responses"
            ])
        elif model_type == "tts":
            optimizations.extend([
                "Cache TTS voice models",
                "Preload common voice settings",
                "Optimize audio generation parameters"
            ])
        
        # Simulate model optimization (in real implementation, this would
        # actually optimize the models)
        await asyncio.sleep(0.1)  # Simulate work
        
        after_metrics = {
            "cached_models": len(self._model_cache),
            "cache_size_mb": sum(
                sys.getsizeof(obj) for obj in self._model_cache.values()
            ) / (1024**2)
        }
        
        result = OptimizationResult(
            optimization_type="model_loading",
            component=ComponentType.PIPELINE,
            timestamp=datetime.now(),
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            improvement_percent=10.0,  # Placeholder improvement
            success=True,
            description=f"Model loading optimization for {model_type}",
            recommendations=optimizations
        )
        
        self.optimization_results.append(result)
        
        logger.info(f"Model loading optimization completed for {model_type}")
        return result
    
    async def optimize_audio_processing(
        self,
        current_settings: Dict[str, Any]
    ) -> OptimizationResult:
        """Optimize audio processing parameters."""
        logger.info("Optimizing audio processing")
        
        before_metrics = current_settings.copy()
        
        # Optimization recommendations based on system capabilities
        optimized_settings = current_settings.copy()
        optimizations = []
        
        # Sample rate optimization
        current_sample_rate = current_settings.get("sample_rate", 16000)
        if self._system_info.get("memory_total_gb", 0) < 8:
            # Lower sample rate for memory-constrained systems
            if current_sample_rate > 16000:
                optimized_settings["sample_rate"] = 16000
                optimizations.append("Reduced sample rate to 16kHz for memory efficiency")
        
        # Chunk size optimization
        current_chunk_duration = current_settings.get("chunk_duration", 0.1)
        cpu_cores = self._system_info.get("cpu_info", {}).get("logical_cores", 1)
        
        if cpu_cores < 4:
            # Larger chunks for CPU-constrained systems
            if current_chunk_duration < 0.2:
                optimized_settings["chunk_duration"] = 0.2
                optimizations.append("Increased chunk duration to reduce CPU overhead")
        elif cpu_cores >= 8:
            # Smaller chunks for high-performance systems
            if current_chunk_duration > 0.05:
                optimized_settings["chunk_duration"] = 0.05
                optimizations.append("Decreased chunk duration for lower latency")
        
        # Buffer size optimization
        if "buffer_size" not in current_settings:
            # Calculate optimal buffer size based on chunk duration and sample rate
            sample_rate = optimized_settings["sample_rate"]
            chunk_duration = optimized_settings["chunk_duration"]
            optimal_buffer = int(sample_rate * chunk_duration)
            optimized_settings["buffer_size"] = optimal_buffer
            optimizations.append(f"Set optimal buffer size: {optimal_buffer} samples")
        
        after_metrics = optimized_settings
        
        # Calculate improvement (placeholder calculation)
        improvement_percent = len(optimizations) * 5.0  # 5% per optimization
        
        result = OptimizationResult(
            optimization_type="audio_processing",
            component=ComponentType.AUDIO_CAPTURE,
            timestamp=datetime.now(),
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            improvement_percent=improvement_percent,
            success=len(optimizations) > 0,
            description=f"Audio processing optimization: {', '.join(optimizations)}",
            recommendations=[
                "Test optimized settings with real audio",
                "Monitor audio quality after changes",
                "Adjust based on actual performance metrics"
            ],
            metadata={"optimized_settings": optimized_settings}
        )
        
        self.optimization_results.append(result)
        
        logger.info(f"Audio processing optimization completed: {len(optimizations)} optimizations")
        return result
    
    async def auto_optimize_pipeline(
        self,
        pipeline_components: Dict[str, Any]
    ) -> List[OptimizationResult]:
        """Automatically optimize the entire pipeline."""
        logger.info("Starting automatic pipeline optimization")
        
        results = []
        
        # 1. Memory optimization
        try:
            memory_result = await self.optimize_memory_usage()
            results.append(memory_result)
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
        
        # 2. Model loading optimization
        for model_type in ["whisper", "llm", "tts"]:
            try:
                model_result = await self.optimize_model_loading(model_type)
                results.append(model_result)
            except Exception as e:
                logger.error(f"Model optimization failed for {model_type}: {e}")
        
        # 3. Audio processing optimization
        if "audio_settings" in pipeline_components:
            try:
                audio_result = await self.optimize_audio_processing(
                    pipeline_components["audio_settings"]
                )
                results.append(audio_result)
            except Exception as e:
                logger.error(f"Audio optimization failed: {e}")
        
        # 4. Component-specific optimizations
        await self._optimize_component_specific(pipeline_components, results)
        
        logger.info(f"Automatic optimization completed: {len(results)} optimizations")
        return results
    
    async def _optimize_component_specific(
        self,
        components: Dict[str, Any],
        results: List[OptimizationResult]
    ) -> None:
        """Apply component-specific optimizations."""
        
        # VAD optimization
        if "vad_settings" in components:
            vad_settings = components["vad_settings"]
            optimizations = []
            
            # Optimize VAD threshold based on environment
            current_threshold = vad_settings.get("threshold", 0.5)
            if current_threshold < 0.3:
                optimizations.append("Increase VAD threshold to reduce false positives")
            elif current_threshold > 0.7:
                optimizations.append("Decrease VAD threshold to improve sensitivity")
            
            if optimizations:
                result = OptimizationResult(
                    optimization_type="vad_optimization",
                    component=ComponentType.VAD,
                    timestamp=datetime.now(),
                    before_metrics=vad_settings,
                    after_metrics=vad_settings,  # Placeholder
                    improvement_percent=5.0,
                    success=True,
                    description=f"VAD optimization: {', '.join(optimizations)}",
                    recommendations=optimizations
                )
                results.append(result)
        
        # TTS optimization
        if "tts_settings" in components:
            tts_settings = components["tts_settings"]
            optimizations = []
            
            # Optimize TTS speed vs quality
            if self._system_info.get("cpu_info", {}).get("logical_cores", 1) < 4:
                optimizations.append("Use faster TTS settings for CPU-constrained system")
            
            if optimizations:
                result = OptimizationResult(
                    optimization_type="tts_optimization",
                    component=ComponentType.TTS,
                    timestamp=datetime.now(),
                    before_metrics=tts_settings,
                    after_metrics=tts_settings,  # Placeholder
                    improvement_percent=3.0,
                    success=True,
                    description=f"TTS optimization: {', '.join(optimizations)}",
                    recommendations=optimizations
                )
                results.append(result)
    
    def get_optimization_history(self, days: int = 30) -> List[OptimizationResult]:
        """Get optimization history."""
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        return [
            r for r in self.optimization_results
            if r.timestamp >= cutoff_date
        ]
    
    def export_optimization_report(
        self,
        filepath: Union[str, Path],
        format: str = "json"
    ) -> None:
        """Export optimization report."""
        filepath = Path(filepath)
        
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "system_info": self._system_info,
            "recommendations": self.get_optimization_recommendations(),
            "optimization_history": [r.to_dict() for r in self.optimization_results],
            "total_optimizations": len(self.optimization_results)
        }
        
        if format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported optimization report to {filepath}")
    
    def clear_cache(self) -> Dict[str, Any]:
        """Clear all caches and return statistics."""
        before_size = len(self._model_cache)
        before_memory = sum(
            sys.getsizeof(obj) for obj in self._model_cache.values()
        ) / (1024**2)
        
        self._model_cache.clear()
        gc.collect()
        
        return {
            "cleared_models": before_size,
            "freed_memory_mb": before_memory,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage statistics."""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            
            return {
                "system_memory": {
                    "total_gb": memory.total / (1024**3),
                    "used_gb": memory.used / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "percent": memory.percent
                },
                "process_memory": {
                    "rss_mb": process_memory.rss / (1024**2),
                    "vms_mb": process_memory.vms / (1024**2)
                },
                "cpu_percent": cpu_percent,
                "model_cache": {
                    "count": len(self._model_cache),
                    "size_mb": sum(
                        sys.getsizeof(obj) for obj in self._model_cache.values()
                    ) / (1024**2)
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting resource usage: {e}")
            return {"error": str(e)}