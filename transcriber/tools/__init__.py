"""
Tool system initialization and registry setup.
"""

from .base import (
    BaseTool, ToolCategory, ToolMetadata, ToolParameter,
    ToolPermission, ToolResult
)
from .registry import ToolRegistry, ToolDiscovery, get_registry, discover_tools


def initialize_tools():
    """Initialize the tool system and discover all available tools."""
    discovered = discover_tools()
    return discovered


__all__ = [
    'BaseTool',
    'ToolCategory',
    'ToolMetadata',
    'ToolParameter',
    'ToolPermission',
    'ToolResult',
    'ToolRegistry',
    'ToolDiscovery',
    'get_registry',
    'discover_tools',
    'initialize_tools'
]