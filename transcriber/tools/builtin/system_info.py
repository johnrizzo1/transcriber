"""
System information and management tools.
"""

import os
import platform
import psutil
import socket
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..base import BaseTool, ToolCategory, ToolMetadata, ToolParameter, ToolPermission


class SystemInfoTool(BaseTool):
    """Tool for getting system information."""
    
    def _define_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="system_info",
            description="Get detailed system information including OS, CPU, memory, and disk usage",
            category=ToolCategory.INFORMATION,
            version="1.0.0",
            author="System",
            permissions=[ToolPermission.READ],
            examples=[
                'system_info()',
                'system_info(include_network=true)',
                'system_info(include_processes=true)'
            ]
        )
    
    def _define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="include_network",
                type="bool",
                description="Include network interface information",
                required=False,
                default=False
            ),
            ToolParameter(
                name="include_processes",
                type="bool",
                description="Include top processes by CPU and memory",
                required=False,
                default=False
            )
        ]
    
    async def _execute(
        self, 
        include_network: bool = False,
        include_processes: bool = False
    ) -> Dict[str, Any]:
        """Get system information."""
        info = {
            "platform": self._get_platform_info(),
            "cpu": self._get_cpu_info(),
            "memory": self._get_memory_info(),
            "disk": self._get_disk_info(),
            "system_time": datetime.now().isoformat()
        }
        
        if include_network:
            info["network"] = self._get_network_info()
        
        if include_processes:
            info["processes"] = self._get_process_info()
        
        return info
    
    def _get_platform_info(self) -> Dict[str, Any]:
        """Get platform/OS information."""
        return {
            "system": platform.system(),
            "node": platform.node(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version()
        }
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        cpu_freq = psutil.cpu_freq()
        cpu_stats = psutil.cpu_stats()
        
        return {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "usage_percent": psutil.cpu_percent(interval=1),
            "per_cpu_usage": psutil.cpu_percent(interval=1, percpu=True),
            "frequency": {
                "current": cpu_freq.current if cpu_freq else None,
                "min": cpu_freq.min if cpu_freq else None,
                "max": cpu_freq.max if cpu_freq else None
            },
            "stats": {
                "ctx_switches": cpu_stats.ctx_switches,
                "interrupts": cpu_stats.interrupts,
                "soft_interrupts": cpu_stats.soft_interrupts,
                "syscalls": cpu_stats.syscalls
            }
        }
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information."""
        virtual_mem = psutil.virtual_memory()
        swap_mem = psutil.swap_memory()
        
        return {
            "virtual": {
                "total": virtual_mem.total,
                "available": virtual_mem.available,
                "used": virtual_mem.used,
                "free": virtual_mem.free,
                "percent": virtual_mem.percent,
                "total_gb": round(virtual_mem.total / (1024**3), 2),
                "available_gb": round(virtual_mem.available / (1024**3), 2),
                "used_gb": round(virtual_mem.used / (1024**3), 2)
            },
            "swap": {
                "total": swap_mem.total,
                "used": swap_mem.used,
                "free": swap_mem.free,
                "percent": swap_mem.percent,
                "sin": swap_mem.sin,
                "sout": swap_mem.sout
            }
        }
    
    def _get_disk_info(self) -> Dict[str, Any]:
        """Get disk usage information."""
        partitions = []
        
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                partitions.append({
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "fstype": partition.fstype,
                    "opts": partition.opts,
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free,
                    "percent": usage.percent,
                    "total_gb": round(usage.total / (1024**3), 2),
                    "used_gb": round(usage.used / (1024**3), 2),
                    "free_gb": round(usage.free / (1024**3), 2)
                })
            except PermissionError:
                # Some partitions may not be accessible
                continue
        
        # Get disk I/O statistics
        io_counters = psutil.disk_io_counters()
        
        return {
            "partitions": partitions,
            "io_counters": {
                "read_count": io_counters.read_count,
                "write_count": io_counters.write_count,
                "read_bytes": io_counters.read_bytes,
                "write_bytes": io_counters.write_bytes,
                "read_time": io_counters.read_time,
                "write_time": io_counters.write_time
            } if io_counters else None
        }
    
    def _get_network_info(self) -> Dict[str, Any]:
        """Get network interface information."""
        interfaces = {}
        
        # Get network interfaces
        for interface, addrs in psutil.net_if_addrs().items():
            interface_info = {"addresses": []}
            
            for addr in addrs:
                addr_info = {
                    "family": addr.family.name,
                    "address": addr.address,
                    "netmask": addr.netmask,
                    "broadcast": addr.broadcast
                }
                interface_info["addresses"].append(addr_info)
            
            # Get interface stats
            stats = psutil.net_if_stats().get(interface)
            if stats:
                interface_info["stats"] = {
                    "isup": stats.isup,
                    "duplex": stats.duplex.name if stats.duplex else None,
                    "speed": stats.speed,
                    "mtu": stats.mtu
                }
            
            interfaces[interface] = interface_info
        
        # Get overall network I/O
        io_counters = psutil.net_io_counters()
        
        return {
            "interfaces": interfaces,
            "hostname": socket.gethostname(),
            "io_counters": {
                "bytes_sent": io_counters.bytes_sent,
                "bytes_recv": io_counters.bytes_recv,
                "packets_sent": io_counters.packets_sent,
                "packets_recv": io_counters.packets_recv,
                "errin": io_counters.errin,
                "errout": io_counters.errout,
                "dropin": io_counters.dropin,
                "dropout": io_counters.dropout
            }
        }
    
    def _get_process_info(self, top_n: int = 10) -> Dict[str, Any]:
        """Get information about running processes."""
        processes = []
        
        # Get all processes
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Sort by CPU usage
        top_cpu = sorted(processes, key=lambda x: x.get('cpu_percent', 0), reverse=True)[:top_n]
        
        # Sort by memory usage
        top_memory = sorted(processes, key=lambda x: x.get('memory_percent', 0), reverse=True)[:top_n]
        
        return {
            "total_processes": len(processes),
            "top_cpu": top_cpu,
            "top_memory": top_memory
        }


class ProcessManagementTool(BaseTool):
    """Tool for managing system processes."""
    
    def _define_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="process_manager",
            description="List, find, and get information about system processes",
            category=ToolCategory.SYSTEM,
            version="1.0.0",
            author="System",
            permissions=[ToolPermission.READ, ToolPermission.EXECUTE],
            examples=[
                'process_manager(action="list", limit=20)',
                'process_manager(action="find", name="python")',
                'process_manager(action="info", pid=1234)'
            ]
        )
    
    def _define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="str",
                description="Action to perform",
                required=True,
                choices=["list", "find", "info"]
            ),
            ToolParameter(
                name="name",
                type="str",
                description="Process name to search for (for 'find' action)",
                required=False,
                default=None
            ),
            ToolParameter(
                name="pid",
                type="int",
                description="Process ID (for 'info' action)",
                required=False,
                default=None
            ),
            ToolParameter(
                name="limit",
                type="int",
                description="Maximum number of processes to return",
                required=False,
                default=50
            ),
            ToolParameter(
                name="sort_by",
                type="str",
                description="Field to sort by",
                required=False,
                default="cpu_percent",
                choices=["cpu_percent", "memory_percent", "pid", "name", "create_time"]
            )
        ]
    
    async def _execute(
        self,
        action: str,
        name: Optional[str] = None,
        pid: Optional[int] = None,
        limit: int = 50,
        sort_by: str = "cpu_percent"
    ) -> Any:
        """Execute process management action."""
        if action == "list":
            return self._list_processes(limit, sort_by)
        elif action == "find":
            if not name:
                raise ValueError("Process name required for 'find' action")
            return self._find_processes(name, limit)
        elif action == "info":
            if not pid:
                raise ValueError("Process ID required for 'info' action")
            return self._get_process_info(pid)
    
    def _list_processes(self, limit: int, sort_by: str) -> List[Dict[str, Any]]:
        """List running processes."""
        processes = []
        
        attrs = ['pid', 'name', 'cpu_percent', 'memory_percent', 'status', 
                'create_time', 'username', 'cmdline']
        
        for proc in psutil.process_iter(attrs):
            try:
                info = proc.info
                # Get CPU percent with a short interval
                info['cpu_percent'] = proc.cpu_percent(interval=0.1)
                processes.append(info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Sort processes
        reverse = sort_by in ['cpu_percent', 'memory_percent', 'create_time']
        processes.sort(key=lambda x: x.get(sort_by, 0), reverse=reverse)
        
        return processes[:limit]
    
    def _find_processes(self, name: str, limit: int) -> List[Dict[str, Any]]:
        """Find processes by name."""
        matching = []
        name_lower = name.lower()
        
        attrs = ['pid', 'name', 'cpu_percent', 'memory_percent', 'status', 
                'create_time', 'username', 'cmdline']
        
        for proc in psutil.process_iter(attrs):
            try:
                proc_info = proc.info
                if name_lower in proc_info['name'].lower():
                    proc_info['cpu_percent'] = proc.cpu_percent(interval=0.1)
                    matching.append(proc_info)
                    
                    if len(matching) >= limit:
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return matching
    
    def _get_process_info(self, pid: int) -> Dict[str, Any]:
        """Get detailed information about a specific process."""
        try:
            proc = psutil.Process(pid)
            
            # Get all available information
            info = {
                "pid": proc.pid,
                "name": proc.name(),
                "exe": proc.exe() if hasattr(proc, 'exe') else None,
                "cmdline": proc.cmdline(),
                "status": proc.status(),
                "username": proc.username(),
                "create_time": datetime.fromtimestamp(proc.create_time()).isoformat(),
                "cpu_percent": proc.cpu_percent(interval=0.5),
                "memory_percent": proc.memory_percent(),
                "memory_info": proc.memory_info()._asdict(),
                "num_threads": proc.num_threads(),
                "num_fds": proc.num_fds() if hasattr(proc, 'num_fds') else None,
                "connections": len(proc.connections()) if hasattr(proc, 'connections') else None,
                "nice": proc.nice() if hasattr(proc, 'nice') else None,
                "ionice": proc.ionice()._asdict() if hasattr(proc, 'ionice') else None,
                "cpu_affinity": proc.cpu_affinity() if hasattr(proc, 'cpu_affinity') else None,
                "cpu_times": proc.cpu_times()._asdict(),
                "io_counters": proc.io_counters()._asdict() if hasattr(proc, 'io_counters') else None
            }
            
            # Get parent and children
            try:
                parent = proc.parent()
                info["parent"] = {"pid": parent.pid, "name": parent.name()} if parent else None
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                info["parent"] = None
            
            try:
                children = proc.children()
                info["children"] = [{"pid": c.pid, "name": c.name()} for c in children]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                info["children"] = []
            
            return info
            
        except psutil.NoSuchProcess:
            raise ValueError(f"No process found with PID {pid}")
        except psutil.AccessDenied:
            raise PermissionError(f"Access denied to process {pid}")


class EnvironmentTool(BaseTool):
    """Tool for accessing and managing environment variables."""
    
    def _define_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="environment",
            description="Access and manage environment variables",
            category=ToolCategory.SYSTEM,
            version="1.0.0",
            author="System",
            permissions=[ToolPermission.READ],
            examples=[
                'environment(action="get", name="PATH")',
                'environment(action="list")',
                'environment(action="list", filter="PYTHON")'
            ]
        )
    
    def _define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="str",
                description="Action to perform",
                required=True,
                choices=["get", "list"]
            ),
            ToolParameter(
                name="name",
                type="str",
                description="Environment variable name (for 'get' action)",
                required=False,
                default=None
            ),
            ToolParameter(
                name="filter",
                type="str",
                description="Filter pattern for listing variables",
                required=False,
                default=None
            )
        ]
    
    async def _execute(
        self,
        action: str,
        name: Optional[str] = None,
        filter: Optional[str] = None
    ) -> Any:
        """Execute environment variable operation."""
        if action == "get":
            if not name:
                raise ValueError("Variable name required for 'get' action")
            return self._get_env_var(name)
        elif action == "list":
            return self._list_env_vars(filter)
    
    def _get_env_var(self, name: str) -> Dict[str, Any]:
        """Get a specific environment variable."""
        value = os.environ.get(name)
        
        if value is None:
            return {
                "name": name,
                "exists": False,
                "value": None
            }
        
        return {
            "name": name,
            "exists": True,
            "value": value,
            "length": len(value)
        }
    
    def _list_env_vars(self, filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List environment variables."""
        env_vars = []
        
        for name, value in os.environ.items():
            if filter and filter.upper() not in name.upper():
                continue
            
            env_vars.append({
                "name": name,
                "value": value,
                "length": len(value)
            })
        
        # Sort by name
        env_vars.sort(key=lambda x: x["name"])
        
        return env_vars


class UptimeTool(BaseTool):
    """Tool for getting system uptime and boot time."""
    
    def _define_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="uptime",
            description="Get system uptime and boot time information",
            category=ToolCategory.INFORMATION,
            version="1.0.0",
            author="System",
            permissions=[ToolPermission.READ],
            examples=[
                'uptime()',
                'uptime(format="human")',
                'uptime(format="seconds")'
            ]
        )
    
    def _define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="format",
                type="str",
                description="Output format",
                required=False,
                default="human",
                choices=["human", "seconds", "detailed"]
            )
        ]
    
    async def _execute(self, format: str = "human") -> Any:
        """Get system uptime."""
        boot_time = psutil.boot_time()
        current_time = datetime.now().timestamp()
        uptime_seconds = current_time - boot_time
        
        if format == "seconds":
            return {
                "uptime_seconds": int(uptime_seconds),
                "boot_timestamp": boot_time
            }
        elif format == "human":
            return {
                "uptime": self._format_uptime(uptime_seconds),
                "boot_time": datetime.fromtimestamp(boot_time).strftime("%Y-%m-%d %H:%M:%S")
            }
        else:  # detailed
            days = int(uptime_seconds // 86400)
            hours = int((uptime_seconds % 86400) // 3600)
            minutes = int((uptime_seconds % 3600) // 60)
            seconds = int(uptime_seconds % 60)
            
            return {
                "uptime": {
                    "days": days,
                    "hours": hours,
                    "minutes": minutes,
                    "seconds": seconds,
                    "total_seconds": int(uptime_seconds),
                    "human_readable": self._format_uptime(uptime_seconds)
                },
                "boot_time": {
                    "timestamp": boot_time,
                    "datetime": datetime.fromtimestamp(boot_time).isoformat(),
                    "formatted": datetime.fromtimestamp(boot_time).strftime("%Y-%m-%d %H:%M:%S")
                },
                "current_time": {
                    "timestamp": current_time,
                    "datetime": datetime.now().isoformat(),
                    "formatted": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format."""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days} day{'s' if days != 1 else ''}")
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        
        if not parts:
            return "Less than a minute"
        
        return ", ".join(parts)