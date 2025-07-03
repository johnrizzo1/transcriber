"""
Tool registry system for managing and discovering tools.
"""

import asyncio
import importlib
import inspect
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Type

from .base import BaseTool, ToolCategory, ToolPermission, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[ToolCategory, List[str]] = {
            category: [] for category in ToolCategory
        }
        self._permissions_required: Set[ToolPermission] = set()
        self._initialized = False
        
    def register(self, tool: BaseTool) -> None:
        """
        Register a tool in the registry.
        
        Args:
            tool: Tool instance to register
        """
        name = tool.name
        
        if name in self._tools:
            logger.warning(f"Tool '{name}' already registered, overwriting")
        
        self._tools[name] = tool
        
        # Add to category index
        category = tool.metadata.category
        if name not in self._categories[category]:
            self._categories[category].append(name)
        
        # Track required permissions
        self._permissions_required.update(tool.metadata.permissions)
        
        logger.info(f"Registered tool: {name} (category: {category.value})")
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a tool from the registry.
        
        Args:
            name: Tool name to unregister
            
        Returns:
            True if tool was unregistered, False if not found
        """
        if name not in self._tools:
            return False
        
        tool = self._tools[name]
        category = tool.metadata.category
        
        # Remove from registry
        del self._tools[name]
        
        # Remove from category index
        if name in self._categories[category]:
            self._categories[category].remove(name)
        
        logger.info(f"Unregistered tool: {name}")
        return True
    
    def get(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)
    
    def get_all(self) -> Dict[str, BaseTool]:
        """Get all registered tools."""
        return self._tools.copy()
    
    def get_by_category(self, category: ToolCategory) -> List[BaseTool]:
        """
        Get all tools in a specific category.
        
        Args:
            category: Tool category
            
        Returns:
            List of tools in the category
        """
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]
    
    def get_by_permission(self, permission: ToolPermission) -> List[BaseTool]:
        """
        Get all tools that require a specific permission.
        
        Args:
            permission: Permission to filter by
            
        Returns:
            List of tools requiring the permission
        """
        tools = []
        for tool in self._tools.values():
            if permission in tool.metadata.permissions:
                tools.append(tool)
        return tools
    
    def search(self, query: str) -> List[BaseTool]:
        """
        Search for tools by name or description.
        
        Args:
            query: Search query (case-insensitive)
            
        Returns:
            List of matching tools
        """
        query_lower = query.lower()
        results = []
        
        for tool in self._tools.values():
            if (query_lower in tool.name.lower() or 
                query_lower in tool.description.lower()):
                results.append(tool)
        
        return results
    
    def list_tools(self) -> List[str]:
        """Get a list of all tool names."""
        return list(self._tools.keys())
    
    def list_categories(self) -> Dict[str, List[str]]:
        """Get a mapping of categories to tool names."""
        return {
            category.value: self._categories[category]
            for category in ToolCategory
            if self._categories[category]
        }
    
    def get_required_permissions(self) -> Set[ToolPermission]:
        """Get all permissions required by registered tools."""
        return self._permissions_required.copy()
    
    async def execute_tool(
        self, 
        name: str, 
        **kwargs
    ) -> ToolResult:
        """
        Execute a tool by name.
        
        Args:
            name: Tool name
            **kwargs: Tool parameters
            
        Returns:
            ToolResult from execution
        """
        tool = self.get(name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool '{name}' not found",
                metadata={"available_tools": self.list_tools()}
            )
        
        return await tool.execute(**kwargs)
    
    def get_tool_info(self, name: str) -> Optional[Dict[str, any]]:
        """
        Get detailed information about a tool.
        
        Args:
            name: Tool name
            
        Returns:
            Tool information dictionary or None if not found
        """
        tool = self.get(name)
        if not tool:
            return None
        
        return {
            "name": tool.name,
            "description": tool.description,
            "category": tool.metadata.category.value,
            "version": tool.metadata.version,
            "author": tool.metadata.author,
            "permissions": [p.value for p in tool.metadata.permissions],
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                    "choices": p.choices
                }
                for p in tool.parameters
            ],
            "examples": tool.metadata.examples,
            "usage": tool.get_usage_example()
        }
    
    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        for category in self._categories:
            self._categories[category].clear()
        self._permissions_required.clear()
        logger.info("Tool registry cleared")


class ToolDiscovery:
    """Automatic tool discovery system."""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        
    def discover_in_module(self, module_name: str) -> List[str]:
        """
        Discover tools in a Python module.
        
        Args:
            module_name: Module to scan for tools
            
        Returns:
            List of discovered tool names
        """
        discovered = []
        
        try:
            module = importlib.import_module(module_name)
            
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseTool) and 
                    obj is not BaseTool and
                    not inspect.isabstract(obj)):
                    
                    try:
                        # Instantiate and register the tool
                        tool_instance = obj()
                        self.registry.register(tool_instance)
                        discovered.append(tool_instance.name)
                        
                    except Exception as e:
                        logger.error(f"Failed to instantiate tool {name}: {e}")
                        
        except ImportError as e:
            logger.error(f"Failed to import module {module_name}: {e}")
            
        return discovered
    
    def discover_in_directory(self, directory: Path) -> List[str]:
        """
        Discover tools in a directory of Python files.
        
        Args:
            directory: Directory to scan
            
        Returns:
            List of discovered tool names
        """
        discovered = []
        directory = Path(directory)
        
        if not directory.exists() or not directory.is_dir():
            logger.warning(f"Directory {directory} does not exist")
            return discovered
        
        # Find all Python files
        for py_file in directory.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
                
            # Convert file path to module name
            try:
                # Calculate module path relative to package root
                package_root = Path(__file__).parent.parent  # transcriber/
                rel_path = py_file.relative_to(package_root)
                module_name = str(rel_path.with_suffix("")).replace("/", ".").replace("\\", ".")
                module_name = f"transcriber.{module_name}"
                
                # Discover tools in the module
                module_tools = self.discover_in_module(module_name)
                discovered.extend(module_tools)
                
            except Exception as e:
                logger.error(f"Failed to process module {py_file}: {e}")
        
        return discovered
    
    def auto_discover(self) -> List[str]:
        """
        Automatically discover tools in standard locations.
        
        Returns:
            List of discovered tool names
        """
        discovered = []
        
        # Standard tool directories
        tools_dir = Path(__file__).parent / "builtin"
        if tools_dir.exists():
            discovered.extend(self.discover_in_directory(tools_dir))
        
        # User tools directory
        user_tools_dir = Path.home() / ".transcriber" / "tools"
        if user_tools_dir.exists():
            discovered.extend(self.discover_in_directory(user_tools_dir))
        
        logger.info(f"Auto-discovered {len(discovered)} tools")
        return discovered


# Global registry instance
tool_registry = ToolRegistry()
tool_discovery = ToolDiscovery(tool_registry)


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return tool_registry


def discover_tools() -> List[str]:
    """Discover and register all available tools."""
    return tool_discovery.auto_discover()