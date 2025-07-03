# Tool Development Guide

Complete guide for creating, testing, and integrating custom tools into the AI Voice Agent.

## Table of Contents

1. [Overview](#overview)
2. [Tool Architecture](#tool-architecture)
3. [Creating Your First Tool](#creating-your-first-tool)
4. [Tool Metadata](#tool-metadata)
5. [Parameter Definition](#parameter-definition)
6. [Tool Implementation](#tool-implementation)
7. [Error Handling](#error-handling)
8. [Testing Tools](#testing-tools)
9. [Advanced Features](#advanced-features)
10. [Best Practices](#best-practices)
11. [Examples](#examples)
12. [Deployment](#deployment)

## Overview

The AI Voice Agent uses an extensible tool system that allows developers to add custom functionality. Tools are Python classes that inherit from `BaseTool` and implement specific methods for metadata, parameters, and execution logic.

### Key Concepts

- **Tool**: A discrete piece of functionality that the AI can invoke
- **Registry**: Central system that manages and executes tools
- **Metadata**: Descriptive information about the tool
- **Parameters**: Input specification for the tool
- **Permissions**: Security controls for tool execution

### Tool Categories

```python
from transcriber.tools.base import ToolCategory

# Available categories
ToolCategory.SYSTEM      # System operations (files, processes)
ToolCategory.UTILITY     # General utilities (calculations, conversions)
ToolCategory.INFORMATION # Information retrieval (web, databases)
ToolCategory.PRODUCTIVITY # Task management, scheduling
ToolCategory.DEVELOPMENT # Development tools (git, testing)
ToolCategory.CUSTOM      # User-defined tools
```

## Tool Architecture

### Base Tool Structure

```python
from transcriber.tools.base import BaseTool, ToolMetadata, ToolParameter
from typing import List, Any

class MyTool(BaseTool):
    def _define_metadata(self) -> ToolMetadata:
        """Define tool metadata"""
        pass
    
    def _define_parameters(self) -> List[ToolParameter]:
        """Define tool parameters"""
        pass
    
    async def _execute(self, **kwargs) -> Any:
        """Execute tool logic"""
        pass
```

### Tool Lifecycle

1. **Registration**: Tool is registered with the registry
2. **Discovery**: AI discovers available tools
3. **Validation**: Parameters are validated
4. **Execution**: Tool logic is executed
5. **Response**: Results are returned to the AI

## Creating Your First Tool

Let's create a simple "Hello World" tool:

```python
from transcriber.tools.base import BaseTool, ToolMetadata, ToolParameter, ToolCategory
from typing import List, Any

class HelloWorldTool(BaseTool):
    def _define_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="hello_world",
            description="A simple greeting tool",
            category=ToolCategory.UTILITY,
            version="1.0.0",
            author="Your Name",
            examples=[
                "Say hello to someone",
                "hello_world(name='Alice')"
            ]
        )
    
    def _define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="name",
                type="str",
                description="Name of the person to greet",
                required=True
            ),
            ToolParameter(
                name="language",
                type="str",
                description="Language for greeting",
                required=False,
                default="english",
                choices=["english", "spanish", "french"]
            )
        ]
    
    async def _execute(self, name: str, language: str = "english") -> str:
        greetings = {
            "english": f"Hello, {name}!",
            "spanish": f"¡Hola, {name}!",
            "french": f"Bonjour, {name}!"
        }
        
        return greetings.get(language, greetings["english"])
```

### Registering the Tool

```python
from transcriber.tools.registry import get_registry

# Get the global registry
registry = get_registry()

# Register your tool
registry.register(HelloWorldTool())

# Verify registration
tool = registry.get("hello_world")
print(f"Tool registered: {tool.metadata.name}")
```

## Tool Metadata

The `ToolMetadata` class defines descriptive information about your tool:

```python
from transcriber.tools.base import ToolMetadata, ToolCategory, ToolPermission

metadata = ToolMetadata(
    name="my_tool",                    # Unique identifier
    description="What the tool does",  # Clear description
    category=ToolCategory.UTILITY,     # Tool category
    version="1.0.0",                   # Version string
    author="Your Name",                # Author information
    permissions=[                      # Required permissions
        ToolPermission.READ,
        ToolPermission.NETWORK
    ],
    examples=[                         # Usage examples
        "Example usage description",
        "my_tool(param='value')"
    ],
    tags=["tag1", "tag2"],            # Optional tags
    documentation_url="https://...",   # Optional docs link
    deprecated=False,                  # Deprecation flag
    experimental=False                 # Experimental flag
)
```

### Permission System

```python
from transcriber.tools.base import ToolPermission

# Available permissions
ToolPermission.READ        # Read files/data
ToolPermission.WRITE       # Write files/data
ToolPermission.EXECUTE     # Execute commands
ToolPermission.NETWORK     # Network access
ToolPermission.SYSTEM      # System information
ToolPermission.DANGEROUS   # Potentially dangerous operations
```

## Parameter Definition

Define tool parameters using the `ToolParameter` class:

```python
from transcriber.tools.base import ToolParameter

# Basic parameter
param = ToolParameter(
    name="input_text",
    type="str",
    description="Text to process",
    required=True
)

# Parameter with default value
param_with_default = ToolParameter(
    name="format",
    type="str",
    description="Output format",
    required=False,
    default="json"
)

# Parameter with choices
param_with_choices = ToolParameter(
    name="mode",
    type="str",
    description="Processing mode",
    required=True,
    choices=["fast", "accurate", "balanced"]
)

# Numeric parameter with constraints
numeric_param = ToolParameter(
    name="count",
    type="int",
    description="Number of items",
    required=False,
    default=10,
    min_value=1,
    max_value=100
)

# Complex parameter with validation
complex_param = ToolParameter(
    name="config",
    type="dict",
    description="Configuration object",
    required=False,
    schema={
        "type": "object",
        "properties": {
            "enabled": {"type": "boolean"},
            "threshold": {"type": "number", "minimum": 0, "maximum": 1}
        }
    }
)
```

### Parameter Types

Supported parameter types:
- `str`: String values
- `int`: Integer values
- `float`: Floating-point values
- `bool`: Boolean values
- `list`: List of values
- `dict`: Dictionary/object values
- `file`: File path or content
- `url`: URL string

## Tool Implementation

### Basic Implementation

```python
class FileReaderTool(BaseTool):
    def _define_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="read_file",
            description="Read contents of a text file",
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.READ]
        )
    
    def _define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="file_path",
                type="str",
                description="Path to the file to read",
                required=True
            ),
            ToolParameter(
                name="encoding",
                type="str",
                description="File encoding",
                required=False,
                default="utf-8"
            )
        ]
    
    async def _execute(self, file_path: str, encoding: str = "utf-8") -> str:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            
            return f"File content ({len(content)} characters):\n{content}"
        
        except FileNotFoundError:
            raise ToolExecutionError(f"File not found: {file_path}")
        except PermissionError:
            raise ToolExecutionError(f"Permission denied: {file_path}")
        except UnicodeDecodeError:
            raise ToolExecutionError(f"Cannot decode file with {encoding} encoding")
```

### Async Implementation

```python
import aiohttp
import asyncio

class WebScraperTool(BaseTool):
    def _define_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="web_scraper",
            description="Fetch content from a web page",
            category=ToolCategory.INFORMATION,
            permissions=[ToolPermission.NETWORK]
        )
    
    def _define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="url",
                type="url",
                description="URL to scrape",
                required=True
            ),
            ToolParameter(
                name="timeout",
                type="int",
                description="Request timeout in seconds",
                required=False,
                default=30,
                min_value=1,
                max_value=300
            )
        ]
    
    async def _execute(self, url: str, timeout: int = 30) -> str:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=timeout) as response:
                    if response.status == 200:
                        content = await response.text()
                        return f"Successfully scraped {url}\nContent length: {len(content)} characters"
                    else:
                        raise ToolExecutionError(f"HTTP {response.status}: {response.reason}")
        
        except asyncio.TimeoutError:
            raise ToolExecutionError(f"Request timeout after {timeout} seconds")
        except aiohttp.ClientError as e:
            raise ToolExecutionError(f"Network error: {str(e)}")
```

### Stateful Tools

```python
class DatabaseTool(BaseTool):
    def __init__(self):
        super().__init__()
        self._connection = None
    
    async def _initialize(self):
        """Called when tool is first used"""
        if not self._connection:
            self._connection = await self._create_connection()
    
    async def _cleanup(self):
        """Called when tool is no longer needed"""
        if self._connection:
            await self._connection.close()
            self._connection = None
    
    def _define_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="database_query",
            description="Execute database queries",
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.READ, ToolPermission.WRITE]
        )
    
    def _define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="str",
                description="SQL query to execute",
                required=True
            )
        ]
    
    async def _execute(self, query: str) -> str:
        await self._initialize()
        
        try:
            result = await self._connection.execute(query)
            return f"Query executed successfully. Rows affected: {result.rowcount}"
        except Exception as e:
            raise ToolExecutionError(f"Database error: {str(e)}")
    
    async def _create_connection(self):
        # Implementation specific to your database
        pass
```

## Error Handling

### Tool-Specific Exceptions

```python
from transcriber.tools.base import ToolExecutionError, ToolValidationError

class MyTool(BaseTool):
    async def _execute(self, **kwargs) -> Any:
        try:
            # Tool logic here
            result = self.process_data(kwargs)
            return result
        
        except ValueError as e:
            # Convert to tool-specific error
            raise ToolValidationError(f"Invalid input: {str(e)}")
        
        except ConnectionError as e:
            # Network-related error
            raise ToolExecutionError(f"Connection failed: {str(e)}")
        
        except Exception as e:
            # Generic error handling
            raise ToolExecutionError(f"Unexpected error: {str(e)}")
```

### Validation

```python
class ValidatedTool(BaseTool):
    def _validate_parameters(self, **kwargs) -> None:
        """Custom parameter validation"""
        super()._validate_parameters(**kwargs)
        
        # Custom validation logic
        if 'email' in kwargs:
            email = kwargs['email']
            if '@' not in email:
                raise ToolValidationError("Invalid email format")
        
        if 'age' in kwargs:
            age = kwargs['age']
            if age < 0 or age > 150:
                raise ToolValidationError("Age must be between 0 and 150")
    
    async def _execute(self, **kwargs) -> Any:
        # Parameters are already validated when this is called
        return "Validation passed!"
```

## Testing Tools

### Unit Testing

```python
import pytest
from transcriber.tools.registry import ToolRegistry
from my_tools import HelloWorldTool

class TestHelloWorldTool:
    def setup_method(self):
        self.tool = HelloWorldTool()
        self.registry = ToolRegistry()
        self.registry.register(self.tool)
    
    def test_metadata(self):
        """Test tool metadata"""
        metadata = self.tool.metadata
        assert metadata.name == "hello_world"
        assert metadata.description is not None
        assert metadata.category is not None
    
    def test_parameters(self):
        """Test parameter definition"""
        params = self.tool.parameters
        assert len(params) == 2
        
        name_param = next(p for p in params if p.name == "name")
        assert name_param.required is True
        assert name_param.type == "str"
    
    @pytest.mark.asyncio
    async def test_execution(self):
        """Test tool execution"""
        result = await self.tool.execute(name="Alice")
        assert "Hello, Alice!" in result
    
    @pytest.mark.asyncio
    async def test_execution_with_language(self):
        """Test tool execution with optional parameter"""
        result = await self.tool.execute(name="Bob", language="spanish")
        assert "¡Hola, Bob!" in result
    
    @pytest.mark.asyncio
    async def test_validation_error(self):
        """Test parameter validation"""
        with pytest.raises(ToolValidationError):
            await self.tool.execute()  # Missing required parameter
    
    @pytest.mark.asyncio
    async def test_registry_integration(self):
        """Test tool registry integration"""
        result = await self.registry.execute_tool("hello_world", name="Charlie")
        assert "Hello, Charlie!" in result
```

### Integration Testing

```python
import pytest
from transcriber.agent.core import VoiceAgent
from transcriber.config import settings
from my_tools import MyCustomTool

class TestToolIntegration:
    @pytest.mark.asyncio
    async def test_tool_with_agent(self):
        """Test tool integration with the voice agent"""
        # Register custom tool
        from transcriber.tools.registry import get_registry
        registry = get_registry()
        registry.register(MyCustomTool())
        
        # Initialize agent
        agent = VoiceAgent(settings)
        await agent.initialize()
        
        try:
            # Test tool usage through agent
            response = await agent.process_text("Use my custom tool with input 'test'")
            assert "test" in response.lower()
        
        finally:
            await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_tool_error_handling(self):
        """Test error handling in tool execution"""
        agent = VoiceAgent(settings)
        await agent.initialize()
        
        try:
            # Test with invalid input that should trigger error
            response = await agent.process_text("Use my tool with invalid input")
            # Agent should handle the error gracefully
            assert "error" in response.lower() or "sorry" in response.lower()
        
        finally:
            await agent.cleanup()
```

### Performance Testing

```python
import time
import asyncio
from transcriber.performance.integration import monitor_performance

class TestToolPerformance:
    @pytest.mark.asyncio
    async def test_tool_performance(self):
        """Test tool execution performance"""
        tool = MyCustomTool()
        
        # Measure execution time
        start_time = time.time()
        result = await tool.execute(input="performance test")
        end_time = time.time()
        
        execution_time = end_time - start_time
        assert execution_time < 1.0  # Should complete within 1 second
    
    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test concurrent tool execution"""
        tool = MyCustomTool()
        
        # Run multiple concurrent executions
        tasks = [
            tool.execute(input=f"test_{i}")
            for i in range(10)
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # All should complete successfully
        assert len(results) == 10
        assert all("test_" in str(result) for result in results)
        
        # Should be faster than sequential execution
        total_time = end_time - start_time
        assert total_time < 5.0  # Reasonable concurrent performance
```

## Advanced Features

### Tool Composition

```python
class ComposedTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.registry = get_registry()
    
    async def _execute(self, **kwargs) -> Any:
        # Use other tools within this tool
        file_content = await self.registry.execute_tool(
            "read_file", 
            file_path=kwargs["file_path"]
        )
        
        processed_content = await self.registry.execute_tool(
            "text_processor",
            text=file_content,
            operation="summarize"
        )
        
        return processed_content
```

### Streaming Results

```python
from typing import AsyncIterator

class StreamingTool(BaseTool):
    async def _execute_stream(self, **kwargs) -> AsyncIterator[str]:
        """Stream results as they become available"""
        for i in range(10):
            await asyncio.sleep(0.1)  # Simulate processing
            yield f"Processing step {i+1}/10\n"
        
        yield "Processing complete!"
    
    async def _execute(self, **kwargs) -> str:
        """Fallback for non-streaming execution"""
        results = []
        async for chunk in self._execute_stream(**kwargs):
            results.append(chunk)
        return "".join(results)
```

### Configuration-Driven Tools

```python
from transcriber.config import settings

class ConfigurableTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.config = settings.tools.get("my_tool", {})
    
    def _define_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="configurable_tool",
            description=self.config.get("description", "A configurable tool"),
            category=ToolCategory.CUSTOM
        )
    
    async def _execute(self, **kwargs) -> Any:
        # Use configuration values
        max_items = self.config.get("max_items", 100)
        timeout = self.config.get("timeout", 30)
        
        # Tool logic using configuration
        return f"Processed with max_items={max_items}, timeout={timeout}"
```

## Best Practices

### 1. Clear Naming and Documentation

```python
class WellDocumentedTool(BaseTool):
    """
    A well-documented tool that demonstrates best practices.
    
    This tool processes text data and returns formatted results.
    It supports multiple output formats and includes comprehensive
    error handling.
    """
    
    def _define_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="text_processor",  # Clear, descriptive name
            description="Process and format text data with multiple output options",
            category=ToolCategory.UTILITY,
            examples=[
                "Process text and return as JSON",
                "text_processor(text='Hello World', format='json')",
                "Convert text to uppercase",
                "text_processor(text='hello', operation='uppercase')"
            ]
        )
```

### 2. Robust Error Handling

```python
class RobustTool(BaseTool):
    async def _execute(self, **kwargs) -> Any:
        try:
            # Main logic
            return self._process_data(kwargs)
        
        except FileNotFoundError as e:
            raise ToolExecutionError(f"Required file not found: {e.filename}")
        
        except PermissionError as e:
            raise ToolExecutionError(f"Permission denied: {str(e)}")
        
        except ValueError as e:
            raise ToolValidationError(f"Invalid input value: {str(e)}")
        
        except Exception as e:
            # Log unexpected errors for debugging
            self.logger.error(f"Unexpected error in {self.metadata.name}: {str(e)}")
            raise ToolExecutionError(f"Tool execution failed: {str(e)}")
```

### 3. Resource Management

```python
class ResourceManagedTool(BaseTool):
    def __init__(self):
        super().__init__()
        self._resources = []
    
    async def _execute(self, **kwargs) -> Any:
        resource = None
        try:
            # Acquire resource
            resource = await self._acquire_resource()
            self._resources.append(resource)
            
            # Use resource
            result = await self._process_with_resource(resource, kwargs)
            return result
        
        finally:
            # Always clean up
            if resource:
                await self._release_resource(resource)
                self._resources.remove(resource)
    
    async def _cleanup(self):
        """Clean up any remaining resources"""
        for resource in self._resources[:]:
            await self._release_resource(resource)
        self._resources.clear()
```

### 4. Performance Optimization

```python
from functools import lru_cache
from transcriber.performance.integration import monitor_performance

class OptimizedTool(BaseTool):
    def __init__(self):
        super().__init__()
        self._cache = {}
    
    @monitor_performance("optimized_tool")
    async def _execute(self, **kwargs) -> Any:
        # Check cache first
        cache_key = self._make_cache_key(kwargs)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Process and cache result
        result = await self._expensive_operation(kwargs)
        self._cache[cache_key] = result
        
        return result
    
    @lru_cache(maxsize=128)
    def _expensive_computation(self, data: str) -> str:
        """Cached expensive computation"""
        # Expensive operation here
        return data.upper()
    
    def _make_cache_key(self, kwargs: dict) -> str:
        """Create cache key from parameters"""
        return "|".join(f"{k}:{v}" for k, v in sorted(kwargs.items()))
```

## Examples

### File Processing Tool

```python
import os
import mimetypes
from pathlib import Path

class FileProcessorTool(BaseTool):
    def _define_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="file_processor",
            description="Process files with various operations",
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.READ, ToolPermission.WRITE],
            examples=[
                "Get file information",
                "file_processor(path='/path/to/file', operation='info')",
                "Copy a file",
                "file_processor(path='/src/file', operation='copy', target='/dst/file')"
            ]
        )
    
    def _define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="str",
                description="Path to the file",
                required=True
            ),
            ToolParameter(
                name="operation",
                type="str",
                description="Operation to perform",
                required=True,
                choices=["info", "copy", "move", "delete", "rename"]
            ),
            ToolParameter(
                name="target",
                type="str",
                description="Target path for copy/move/rename operations",
                required=False
            )
        ]
    
    async def _execute(self, path: str, operation: str, target: str = None) -> str:
        file_path = Path(path)
        
        if not file_path.exists():
            raise ToolExecutionError(f"File does not exist: {path}")
        
        if operation == "info":
            return self._get_file_info(file_path)
        elif operation == "copy":
            return await self._copy_file(file_path, target)
        elif operation == "move":
            return await self._move_file(file_path, target)
        elif operation == "delete":
            return await self._delete_file(file_path)
        elif operation == "rename":
            return await self._rename_file(file_path, target)
        else:
            raise ToolValidationError(f"Unknown operation: {operation}")
    
    def _get_file_info(self, file_path: Path) -> str:
        stat = file_path.stat()
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        info = {
            "name": file_path.name,
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "type": mime_type or "unknown",
            "permissions": oct(stat.st_mode)[-3:]
        }
        
        return f"File Info:\n" + "\n".join(f"{k}: {v}" for k, v in info.items())
    
    async def _copy_file(self, source: Path, target: str) -> str:
        if not target:
            raise ToolValidationError("Target path required for copy operation")
        
        target_path = Path(target)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.copy2(source, target_path)
        
        return f"File copied from {source} to {target_path}"
```

### API Client Tool

```python
import aiohttp
import json
from typing import Dict, Any

class APIClientTool(BaseTool):
    def _define_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="api_client",
            description="Make HTTP API requests",
            category=ToolCategory.INFORMATION,
            permissions=[ToolPermission.NETWORK],
            examples=[
                "GET request to an API",
                "api_client(url='https://api.example.com/data', method='GET')",
                "POST request with data",
                "api_client(url='https://api.example.com/create', method='POST', data={'key': 'value'})"
            ]
        )
    
    def _define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="url",
                type="url",
                description="API endpoint URL",
                required=True
            ),
            ToolParameter(
                name="method",
                type="str",
                description="HTTP method",
                required=False,
                default="GET",
                choices=["GET", "POST", "PUT", "DELETE", "PATCH"]
            ),
            ToolParameter(
                name="headers",
                type="dict",
                description="HTTP headers",
                required=False
            ),
            ToolParameter(
                name="data",
                type="dict",
                description="Request data (for POST/PUT/PATCH)",
                required=False
            ),
            ToolParameter(
                name="timeout",
                type="int",
                description="Request timeout in seconds",
                required=False,
                default=30,
                min_value=1,
                max_value=300
            )
        ]
    
    async def _execute(self, url: str, method: str = "GET", 
                      headers: Dict[str, str] = None, 
                      data: Dict[str, Any] = None,
                      timeout: int = 30) -> str:
        
        headers = headers or {}
        
        try:
            async with aiohttp.ClientSession() as session:
                kwargs = {
                    "url": url,
                    "headers": headers,
                    "timeout": aiohttp.ClientTimeout(total=timeout)
                }
                
                if data and method in ["POST", "PUT", "PATCH"]:
                    kwargs["json"] = data
                    headers.setdefault("Content-Type", "application/json")
                
                async with session.request(method, **kwargs) as response:
                    response_text = await response.text()
                    
                    result = {
                        "status": response.status,
                        "headers": dict(response.headers),
                        "body": response_text
                    }
                    
                    # Try to parse JSON response
                    try:
                        result["json"] = json.loads(response_text)
                    except json.JSONDecodeError:
                        pass
                    
                    return f"API Response ({response.status}):\n{json.dumps(result, indent=2)}"
        
        except aiohttp.ClientTimeout:
            raise ToolExecutionError(f"Request timeout after {timeout} seconds")
        except aiohttp.ClientError as e:
            raise ToolExecutionError(f"HTTP client error: {str(e)}")
        except Exception as e:
            raise ToolExecutionError(f"Unexpected error: {str(e)}")
```

## Deployment

### Package Structure

```
my_tools/
├── __init__.py
├── setup.py
├── README.md
├── requirements.txt
├── my_tools/
│   ├── __init__.py
│   ├── file_tools.py
│   ├── api_tools.py
│   └── utils.py
└── tests/
    ├── __init__.py
    ├── test_file_tools.py
    └── test_api_tools.py
```

### Setup Script

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="my-transcriber-tools",
    version="1.0.0",
    description="Custom tools for AI Voice Agent",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "transcriber>=1.0.0",
        "aiohttp>=3.8.0",
        # Other dependencies
    ],
    entry_points={
        "transcriber.tools": [
            "file_processor = my_tools.file_tools:FileProcessorTool",
            "api_client = my_tools.api_tools:APIClientTool",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
)
```

### Installation

```bash
# Install your tool package
pip install my-transcriber-tools

# Or install in development mode
pip install -e .
```

### Auto-Registration

```python
# my_tools