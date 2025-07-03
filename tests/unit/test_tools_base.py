"""
Unit tests for the tool system base classes and interfaces.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from transcriber.tools.base import (
    BaseTool,
    AsyncTool,
    SyncTool,
    ToolCategory,
    ToolPermission,
    ToolMetadata,
    ToolParameter,
    ToolResult
)


@pytest.mark.unit
class TestToolEnums:
    """Test tool enumeration classes."""
    
    def test_tool_category_values(self):
        """Test ToolCategory enum values."""
        assert ToolCategory.SYSTEM.value == "system"
        assert ToolCategory.DEVELOPMENT.value == "development"
        assert ToolCategory.INFORMATION.value == "information"
        assert ToolCategory.PRODUCTIVITY.value == "productivity"
        assert ToolCategory.UTILITY.value == "utility"
    
    def test_tool_permission_values(self):
        """Test ToolPermission enum values."""
        assert ToolPermission.READ.value == "read"
        assert ToolPermission.WRITE.value == "write"
        assert ToolPermission.EXECUTE.value == "execute"
        assert ToolPermission.NETWORK.value == "network"
        assert ToolPermission.SYSTEM.value == "system"


@pytest.mark.unit
class TestToolMetadata:
    """Test ToolMetadata dataclass."""
    
    def test_basic_metadata_creation(self):
        """Test basic metadata creation."""
        metadata = ToolMetadata(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.UTILITY
        )
        
        assert metadata.name == "test_tool"
        assert metadata.description == "A test tool"
        assert metadata.category == ToolCategory.UTILITY
        assert metadata.version == "1.0.0"
        assert metadata.author == "System"
        assert metadata.permissions == []
        assert metadata.examples == []
    
    def test_metadata_with_permissions(self):
        """Test metadata with permissions."""
        permissions = [ToolPermission.READ, ToolPermission.WRITE]
        metadata = ToolMetadata(
            name="file_tool",
            description="File operations tool",
            category=ToolCategory.SYSTEM,
            permissions=permissions
        )
        
        assert metadata.permissions == permissions
    
    def test_metadata_with_examples(self):
        """Test metadata with examples."""
        examples = ["example1", "example2"]
        metadata = ToolMetadata(
            name="calc_tool",
            description="Calculator tool",
            category=ToolCategory.UTILITY,
            examples=examples
        )
        
        assert metadata.examples == examples


@pytest.mark.unit
class TestToolParameter:
    """Test ToolParameter model."""
    
    def test_required_parameter(self):
        """Test required parameter creation."""
        param = ToolParameter(
            name="input",
            type="str",
            description="Input text",
            required=True
        )
        
        assert param.name == "input"
        assert param.type == "str"
        assert param.description == "Input text"
        assert param.required is True
        assert param.default is None
        assert param.choices is None
    
    def test_optional_parameter_with_default(self):
        """Test optional parameter with default value."""
        param = ToolParameter(
            name="count",
            type="int",
            description="Number of items",
            required=False,
            default=10
        )
        
        assert param.required is False
        assert param.default == 10
    
    def test_parameter_with_choices(self):
        """Test parameter with valid choices."""
        choices = ["option1", "option2", "option3"]
        param = ToolParameter(
            name="mode",
            type="str",
            description="Operation mode",
            choices=choices
        )
        
        assert param.choices == choices
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Valid parameter should work
        ToolParameter(
            name="test",
            type="str",
            description="Test parameter"
        )
        
        # Missing required fields should raise ValidationError
        with pytest.raises(ValidationError):
            ToolParameter(name="test")  # Missing type and description


@pytest.mark.unit
class TestToolResult:
    """Test ToolResult model."""
    
    def test_success_result(self):
        """Test successful tool result."""
        result = ToolResult(
            success=True,
            output="Operation completed",
            metadata={"duration": 0.5}
        )
        
        assert result.success is True
        assert result.output == "Operation completed"
        assert result.error is None
        assert result.metadata == {"duration": 0.5}
    
    def test_error_result(self):
        """Test error tool result."""
        result = ToolResult(
            success=False,
            error="Operation failed",
            metadata={"error_code": 500}
        )
        
        assert result.success is False
        assert result.output is None
        assert result.error == "Operation failed"
        assert result.metadata == {"error_code": 500}
    
    def test_result_validation(self):
        """Test result validation."""
        # Valid result should work
        ToolResult(success=True)
        
        # Missing required field should raise ValidationError
        with pytest.raises(ValidationError):
            ToolResult()  # Missing success field


@pytest.mark.unit
class TestBaseTool:
    """Test BaseTool abstract base class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseTool cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTool()
    
    def test_concrete_tool_implementation(self):
        """Test concrete tool implementation."""
        
        class TestTool(BaseTool):
            def _define_metadata(self):
                return ToolMetadata(
                    name="test_tool",
                    description="A test tool",
                    category=ToolCategory.UTILITY
                )
            
            def _define_parameters(self):
                return [
                    ToolParameter(
                        name="input",
                        type="str",
                        description="Input text",
                        required=True
                    )
                ]
            
            async def _execute(self, **kwargs):
                return f"Processed: {kwargs['input']}"
        
        tool = TestTool()
        
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert len(tool.parameters) == 1
        assert tool.parameters[0].name == "input"
    
    def test_parameter_schema_generation(self):
        """Test parameter schema generation."""
        
        class TestTool(BaseTool):
            def _define_metadata(self):
                return ToolMetadata(
                    name="schema_tool",
                    description="Schema test tool",
                    category=ToolCategory.UTILITY
                )
            
            def _define_parameters(self):
                return [
                    ToolParameter(
                        name="text",
                        type="str",
                        description="Input text",
                        required=True
                    ),
                    ToolParameter(
                        name="count",
                        type="int",
                        description="Repeat count",
                        required=False,
                        default=1
                    ),
                    ToolParameter(
                        name="mode",
                        type="str",
                        description="Processing mode",
                        choices=["fast", "slow"]
                    )
                ]
            
            async def _execute(self, **kwargs):
                return "test"
        
        tool = TestTool()
        schema = tool.get_parameter_schema()
        
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        
        # Check properties
        props = schema["properties"]
        assert "text" in props
        assert "count" in props
        assert "mode" in props
        
        # Check required fields
        assert "text" in schema["required"]
        assert "count" not in schema["required"]  # Has default
        
        # Check property details
        assert props["text"]["type"] == "str"
        assert props["count"]["default"] == 1
        assert props["mode"]["enum"] == ["fast", "slow"]
    
    def test_parameter_validation_success(self):
        """Test successful parameter validation."""
        
        class TestTool(BaseTool):
            def _define_metadata(self):
                return ToolMetadata(
                    name="validation_tool",
                    description="Validation test tool",
                    category=ToolCategory.UTILITY
                )
            
            def _define_parameters(self):
                return [
                    ToolParameter(
                        name="text",
                        type="str",
                        description="Input text",
                        required=True
                    ),
                    ToolParameter(
                        name="count",
                        type="int",
                        description="Count",
                        required=False,
                        default=5
                    )
                ]
            
            async def _execute(self, **kwargs):
                return kwargs
        
        tool = TestTool()
        
        # Valid parameters
        validated = tool.validate_parameters(text="hello", count=10)
        assert validated == {"text": "hello", "count": 10}
        
        # Missing optional parameter should use default
        validated = tool.validate_parameters(text="hello")
        assert validated == {"text": "hello", "count": 5}
    
    def test_parameter_validation_errors(self):
        """Test parameter validation errors."""
        
        class TestTool(BaseTool):
            def _define_metadata(self):
                return ToolMetadata(
                    name="error_tool",
                    description="Error test tool",
                    category=ToolCategory.UTILITY
                )
            
            def _define_parameters(self):
                return [
                    ToolParameter(
                        name="required_param",
                        type="str",
                        description="Required parameter",
                        required=True
                    ),
                    ToolParameter(
                        name="choice_param",
                        type="str",
                        description="Choice parameter",
                        choices=["a", "b", "c"]
                    )
                ]
            
            async def _execute(self, **kwargs):
                return kwargs
        
        tool = TestTool()
        
        # Missing required parameter
        with pytest.raises(ValueError, match="Required parameter"):
            tool.validate_parameters()
        
        # Invalid choice
        with pytest.raises(ValueError, match="Invalid value"):
            tool.validate_parameters(
                required_param="test",
                choice_param="invalid"
            )
    
    @pytest.mark.asyncio
    async def test_tool_execution_success(self):
        """Test successful tool execution."""
        
        class TestTool(BaseTool):
            def _define_metadata(self):
                return ToolMetadata(
                    name="exec_tool",
                    description="Execution test tool",
                    category=ToolCategory.UTILITY
                )
            
            def _define_parameters(self):
                return [
                    ToolParameter(
                        name="input",
                        type="str",
                        description="Input text",
                        required=True
                    )
                ]
            
            async def _execute(self, **kwargs):
                return f"Processed: {kwargs['input']}"
        
        tool = TestTool()
        result = await tool.execute(input="test")
        
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.output == "Processed: test"
        assert result.error is None
        assert result.metadata["tool"] == "exec_tool"
        assert result.metadata["parameters"] == {"input": "test"}
    
    @pytest.mark.asyncio
    async def test_tool_execution_error(self):
        """Test tool execution with error."""
        
        class ErrorTool(BaseTool):
            def _define_metadata(self):
                return ToolMetadata(
                    name="error_tool",
                    description="Error test tool",
                    category=ToolCategory.UTILITY
                )
            
            def _define_parameters(self):
                return []
            
            async def _execute(self, **kwargs):
                raise ValueError("Test error")
        
        tool = ErrorTool()
        result = await tool.execute()
        
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.output is None
        assert result.error == "Test error"
        assert result.metadata["error_type"] == "ValueError"
    
    def test_usage_example_generation(self):
        """Test usage example generation."""
        
        class ExampleTool(BaseTool):
            def _define_metadata(self):
                return ToolMetadata(
                    name="example_tool",
                    description="Example tool",
                    category=ToolCategory.UTILITY,
                    examples=["example_tool(text='hello')"]
                )
            
            def _define_parameters(self):
                return [
                    ToolParameter(
                        name="text",
                        type="str",
                        description="Input text",
                        required=True
                    )
                ]
            
            async def _execute(self, **kwargs):
                return kwargs
        
        tool = ExampleTool()
        
        # Should use provided example
        example = tool.get_usage_example()
        assert example == "example_tool(text='hello')"
    
    def test_auto_generated_usage_example(self):
        """Test auto-generated usage example."""
        
        class AutoExampleTool(BaseTool):
            def _define_metadata(self):
                return ToolMetadata(
                    name="auto_tool",
                    description="Auto example tool",
                    category=ToolCategory.UTILITY
                )
            
            def _define_parameters(self):
                return [
                    ToolParameter(
                        name="text",
                        type="str",
                        description="Input text",
                        required=True
                    ),
                    ToolParameter(
                        name="count",
                        type="int",
                        description="Count",
                        required=True
                    )
                ]
            
            async def _execute(self, **kwargs):
                return kwargs
        
        tool = AutoExampleTool()
        example = tool.get_usage_example()
        
        # Should auto-generate example with required parameters
        assert "auto_tool(" in example
        assert "text=" in example
        assert "count=" in example
    
    def test_string_representations(self):
        """Test string representations of tools."""
        
        class StringTool(BaseTool):
            def _define_metadata(self):
                return ToolMetadata(
                    name="string_tool",
                    description="String representation tool",
                    category=ToolCategory.UTILITY,
                    permissions=[ToolPermission.READ]
                )
            
            def _define_parameters(self):
                return []
            
            async def _execute(self, **kwargs):
                return "test"
        
        tool = StringTool()
        
        # Test __str__
        str_repr = str(tool)
        assert str_repr == "string_tool: String representation tool"
        
        # Test __repr__
        repr_str = repr(tool)
        assert "Tool string_tool" in repr_str
        assert "category=utility" in repr_str
        assert "permissions=['read']" in repr_str


@pytest.mark.unit
class TestSyncTool:
    """Test SyncTool class."""
    
    @pytest.mark.asyncio
    async def test_sync_tool_execution(self):
        """Test synchronous tool execution."""
        
        class TestSyncTool(SyncTool):
            def _define_metadata(self):
                return ToolMetadata(
                    name="sync_tool",
                    description="Sync test tool",
                    category=ToolCategory.UTILITY
                )
            
            def _define_parameters(self):
                return [
                    ToolParameter(
                        name="value",
                        type="int",
                        description="Input value",
                        required=True
                    )
                ]
            
            def _execute_sync(self, **kwargs):
                return kwargs["value"] * 2
        
        tool = TestSyncTool()
        result = await tool.execute(value=5)
        
        assert result.success is True
        assert result.output == 10


@pytest.mark.unit
class TestAsyncTool:
    """Test AsyncTool class."""
    
    @pytest.mark.asyncio
    async def test_async_tool_execution(self):
        """Test asynchronous tool execution."""
        
        class TestAsyncTool(AsyncTool):
            def _define_metadata(self):
                return ToolMetadata(
                    name="async_tool",
                    description="Async test tool",
                    category=ToolCategory.UTILITY
                )
            
            def _define_parameters(self):
                return [
                    ToolParameter(
                        name="delay",
                        type="float",
                        description="Delay in seconds",
                        required=True
                    )
                ]
            
            async def _execute(self, **kwargs):
                await asyncio.sleep(kwargs["delay"])
                return f"Waited {kwargs['delay']} seconds"
        
        tool = TestAsyncTool()
        result = await tool.execute(delay=0.01)
        
        assert result.success is True
        assert result.output == "Waited 0.01 seconds"