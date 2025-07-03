"""
Base tool interface and abstract classes for the tool system.
"""

import abc
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Tool categories for organization."""
    SYSTEM = "system"
    DEVELOPMENT = "development"
    INFORMATION = "information"
    PRODUCTIVITY = "productivity"
    UTILITY = "utility"


class ToolPermission(Enum):
    """Tool permission levels."""
    READ = "read"           # Can read data
    WRITE = "write"         # Can write/modify data
    EXECUTE = "execute"     # Can execute programs
    NETWORK = "network"     # Can access network
    SYSTEM = "system"       # Can modify system settings


@dataclass
class ToolMetadata:
    """Metadata for a tool."""
    name: str
    description: str
    category: ToolCategory
    version: str = "1.0.0"
    author: str = "System"
    permissions: List[ToolPermission] = None
    examples: List[str] = None
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = []
        if self.examples is None:
            self.examples = []


class ToolParameter(BaseModel):
    """Definition of a tool parameter."""
    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (str, int, float, bool, list, dict)")
    description: str = Field(..., description="Parameter description")
    required: bool = Field(True, description="Whether the parameter is required")
    default: Any = Field(None, description="Default value if not required")
    choices: Optional[List[Any]] = Field(None, description="Valid choices for the parameter")


class ToolResult(BaseModel):
    """Result returned by a tool execution."""
    success: bool = Field(..., description="Whether the tool executed successfully")
    output: Any = Field(None, description="The output data from the tool")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BaseTool(abc.ABC):
    """Abstract base class for all tools."""
    
    def __init__(self):
        self._metadata = self._define_metadata()
        self._parameters = self._define_parameters()
        
    @abc.abstractmethod
    def _define_metadata(self) -> ToolMetadata:
        """Define the tool's metadata."""
        pass
    
    @abc.abstractmethod
    def _define_parameters(self) -> List[ToolParameter]:
        """Define the tool's parameters."""
        pass
    
    @abc.abstractmethod
    async def _execute(self, **kwargs) -> Any:
        """
        Execute the tool's main logic.
        
        Args:
            **kwargs: Parameters as defined in _define_parameters()
            
        Returns:
            The tool's output data
        """
        pass
    
    @property
    def metadata(self) -> ToolMetadata:
        """Get the tool's metadata."""
        return self._metadata
    
    @property
    def name(self) -> str:
        """Get the tool's name."""
        return self._metadata.name
    
    @property
    def description(self) -> str:
        """Get the tool's description."""
        return self._metadata.description
    
    @property
    def parameters(self) -> List[ToolParameter]:
        """Get the tool's parameters."""
        return self._parameters
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for the tool's parameters."""
        properties = {}
        required = []
        
        for param in self._parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.choices:
                properties[param.name]["enum"] = param.choices
            if param.default is not None:
                properties[param.name]["default"] = param.default
            if param.required:
                required.append(param.name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    def validate_parameters(self, **kwargs) -> Dict[str, Any]:
        """
        Validate and process input parameters.
        
        Args:
            **kwargs: Input parameters
            
        Returns:
            Validated parameters
            
        Raises:
            ValueError: If parameters are invalid
        """
        validated = {}
        
        # Check required parameters
        for param in self._parameters:
            if param.required and param.name not in kwargs:
                raise ValueError(f"Required parameter '{param.name}' not provided")
            
            if param.name in kwargs:
                value = kwargs[param.name]
                
                # Validate choices
                if param.choices and value not in param.choices:
                    raise ValueError(
                        f"Invalid value for '{param.name}': {value}. "
                        f"Must be one of: {param.choices}"
                    )
                
                # Basic type validation
                expected_type = {
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "list": list,
                    "dict": dict
                }.get(param.type)
                
                if expected_type and not isinstance(value, expected_type):
                    try:
                        # Try to convert
                        value = expected_type(value)
                    except (TypeError, ValueError):
                        raise ValueError(
                            f"Invalid type for '{param.name}': expected {param.type}, "
                            f"got {type(value).__name__}"
                        )
                
                validated[param.name] = value
            elif param.default is not None:
                validated[param.name] = param.default
        
        return validated
    
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with validation and error handling.
        
        Args:
            **kwargs: Tool parameters
            
        Returns:
            ToolResult with execution outcome
        """
        try:
            # Validate parameters
            validated_params = self.validate_parameters(**kwargs)
            
            # Log execution
            logger.info(f"Executing tool '{self.name}' with params: {validated_params}")
            
            # Execute tool logic
            output = await self._execute(**validated_params)
            
            # Return success result
            return ToolResult(
                success=True,
                output=output,
                metadata={
                    "tool": self.name,
                    "parameters": validated_params
                }
            )
            
        except Exception as e:
            # Log error
            logger.error(f"Tool '{self.name}' execution failed: {e}")
            
            # Return error result
            return ToolResult(
                success=False,
                error=str(e),
                metadata={
                    "tool": self.name,
                    "parameters": kwargs,
                    "error_type": type(e).__name__
                }
            )
    
    def get_usage_example(self) -> str:
        """Get a usage example for the tool."""
        if self._metadata.examples:
            return self._metadata.examples[0]
        
        # Generate a basic example
        params = []
        for param in self._parameters:
            if param.required:
                example_value = {
                    "str": '"example"',
                    "int": "42",
                    "float": "3.14",
                    "bool": "true",
                    "list": '["item1", "item2"]',
                    "dict": '{"key": "value"}'
                }.get(param.type, '"value"')
                params.append(f"{param.name}={example_value}")
        
        return f"{self.name}({', '.join(params)})"
    
    def __str__(self) -> str:
        """String representation of the tool."""
        return f"{self.name}: {self.description}"
    
    def __repr__(self) -> str:
        """Detailed representation of the tool."""
        return (
            f"<Tool {self.name} "
            f"category={self._metadata.category.value} "
            f"permissions={[p.value for p in self._metadata.permissions]}>"
        )


class AsyncTool(BaseTool):
    """Base class for async tools (default)."""
    pass


class SyncTool(BaseTool):
    """Base class for synchronous tools."""
    
    @abc.abstractmethod
    def _execute_sync(self, **kwargs) -> Any:
        """
        Execute the tool's main logic synchronously.
        
        Args:
            **kwargs: Parameters as defined in _define_parameters()
            
        Returns:
            The tool's output data
        """
        pass
    
    async def _execute(self, **kwargs) -> Any:
        """Convert sync execution to async."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._execute_sync, **kwargs)