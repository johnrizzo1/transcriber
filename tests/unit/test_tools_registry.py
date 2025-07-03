"""
Unit tests for the tool registry system.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from transcriber.tools.base import (
    BaseTool,
    ToolCategory,
    ToolPermission,
    ToolMetadata,
    ToolParameter,
    ToolResult
)
from transcriber.tools.registry import ToolRegistry, ToolDiscovery


# Test tool implementations for testing
class MockCalculatorTool(BaseTool):
    """Mock calculator tool for testing."""
    
    def _define_metadata(self):
        return ToolMetadata(
            name="calculator",
            description="Basic calculator operations",
            category=ToolCategory.UTILITY,
            permissions=[ToolPermission.READ]
        )
    
    def _define_parameters(self):
        return [
            ToolParameter(
                name="expression",
                type="str",
                description="Mathematical expression",
                required=True
            )
        ]
    
    async def _execute(self, **kwargs):
        # Simple mock calculation
        expr = kwargs["expression"]
        if expr == "2+2":
            return 4
        return f"Result of {expr}"


class MockFileOpsTool(BaseTool):
    """Mock file operations tool for testing."""
    
    def _define_metadata(self):
        return ToolMetadata(
            name="file_ops",
            description="File operations",
            category=ToolCategory.SYSTEM,
            permissions=[ToolPermission.READ, ToolPermission.WRITE]
        )
    
    def _define_parameters(self):
        return [
            ToolParameter(
                name="operation",
                type="str",
                description="File operation",
                required=True,
                choices=["read", "write", "list"]
            ),
            ToolParameter(
                name="path",
                type="str",
                description="File path",
                required=True
            )
        ]
    
    async def _execute(self, **kwargs):
        return f"Performed {kwargs['operation']} on {kwargs['path']}"


@pytest.mark.unit
class TestToolRegistry:
    """Test ToolRegistry class."""
    
    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = ToolRegistry()
        
        assert len(registry._tools) == 0
        assert len(registry._categories) == len(ToolCategory)
        assert len(registry._permissions_required) == 0
        assert registry._initialized is False
        
        # Check all categories are initialized
        for category in ToolCategory:
            assert category in registry._categories
            assert registry._categories[category] == []
    
    def test_register_tool(self):
        """Test tool registration."""
        registry = ToolRegistry()
        tool = MockCalculatorTool()
        
        registry.register(tool)
        
        assert "calculator" in registry._tools
        assert registry._tools["calculator"] is tool
        assert "calculator" in registry._categories[ToolCategory.UTILITY]
        assert ToolPermission.READ in registry._permissions_required
    
    def test_register_duplicate_tool(self):
        """Test registering duplicate tool (should overwrite)."""
        registry = ToolRegistry()
        tool1 = MockCalculatorTool()
        tool2 = MockCalculatorTool()
        
        registry.register(tool1)
        registry.register(tool2)  # Should overwrite
        
        assert len(registry._tools) == 1
        assert registry._tools["calculator"] is tool2
        assert len(registry._categories[ToolCategory.UTILITY]) == 1
    
    def test_unregister_tool(self):
        """Test tool unregistration."""
        registry = ToolRegistry()
        tool = MockCalculatorTool()
        
        registry.register(tool)
        assert "calculator" in registry._tools
        
        # Unregister existing tool
        result = registry.unregister("calculator")
        assert result is True
        assert "calculator" not in registry._tools
        assert "calculator" not in registry._categories[ToolCategory.UTILITY]
        
        # Unregister non-existing tool
        result = registry.unregister("nonexistent")
        assert result is False
    
    def test_get_tool(self):
        """Test getting tool by name."""
        registry = ToolRegistry()
        tool = MockCalculatorTool()
        registry.register(tool)
        
        # Get existing tool
        retrieved = registry.get("calculator")
        assert retrieved is tool
        
        # Get non-existing tool
        retrieved = registry.get("nonexistent")
        assert retrieved is None
    
    def test_get_all_tools(self):
        """Test getting all tools."""
        registry = ToolRegistry()
        calc_tool = MockCalculatorTool()
        file_tool = MockFileOpsTool()
        
        registry.register(calc_tool)
        registry.register(file_tool)
        
        all_tools = registry.get_all()
        
        assert len(all_tools) == 2
        assert all_tools["calculator"] is calc_tool
        assert all_tools["file_ops"] is file_tool
        
        # Should return a copy
        all_tools["new_tool"] = "test"
        assert "new_tool" not in registry._tools
    
    def test_get_by_category(self):
        """Test getting tools by category."""
        registry = ToolRegistry()
        calc_tool = MockCalculatorTool()
        file_tool = MockFileOpsTool()
        
        registry.register(calc_tool)
        registry.register(file_tool)
        
        # Get utility tools
        utility_tools = registry.get_by_category(ToolCategory.UTILITY)
        assert len(utility_tools) == 1
        assert utility_tools[0] is calc_tool
        
        # Get system tools
        system_tools = registry.get_by_category(ToolCategory.SYSTEM)
        assert len(system_tools) == 1
        assert system_tools[0] is file_tool
        
        # Get empty category
        dev_tools = registry.get_by_category(ToolCategory.DEVELOPMENT)
        assert len(dev_tools) == 0
    
    def test_get_by_permission(self):
        """Test getting tools by permission."""
        registry = ToolRegistry()
        calc_tool = MockCalculatorTool()
        file_tool = MockFileOpsTool()
        
        registry.register(calc_tool)
        registry.register(file_tool)
        
        # Get tools with READ permission
        read_tools = registry.get_by_permission(ToolPermission.READ)
        assert len(read_tools) == 2
        assert calc_tool in read_tools
        assert file_tool in read_tools
        
        # Get tools with WRITE permission
        write_tools = registry.get_by_permission(ToolPermission.WRITE)
        assert len(write_tools) == 1
        assert file_tool in write_tools
        
        # Get tools with non-existing permission
        exec_tools = registry.get_by_permission(ToolPermission.EXECUTE)
        assert len(exec_tools) == 0
    
    def test_search_tools(self):
        """Test searching tools."""
        registry = ToolRegistry()
        calc_tool = MockCalculatorTool()
        file_tool = MockFileOpsTool()
        
        registry.register(calc_tool)
        registry.register(file_tool)
        
        # Search by name
        results = registry.search("calc")
        assert len(results) == 1
        assert calc_tool in results
        
        # Search by description
        results = registry.search("operations")
        assert len(results) == 2  # Both have "operations" in description
        
        # Case insensitive search
        results = registry.search("CALCULATOR")
        assert len(results) == 1
        assert calc_tool in results
        
        # No matches
        results = registry.search("nonexistent")
        assert len(results) == 0
    
    def test_list_tools(self):
        """Test listing tool names."""
        registry = ToolRegistry()
        calc_tool = MockCalculatorTool()
        file_tool = MockFileOpsTool()
        
        registry.register(calc_tool)
        registry.register(file_tool)
        
        tool_names = registry.list_tools()
        assert len(tool_names) == 2
        assert "calculator" in tool_names
        assert "file_ops" in tool_names
    
    def test_list_categories(self):
        """Test listing categories with tools."""
        registry = ToolRegistry()
        calc_tool = MockCalculatorTool()
        file_tool = MockFileOpsTool()
        
        registry.register(calc_tool)
        registry.register(file_tool)
        
        categories = registry.list_categories()
        
        assert "utility" in categories
        assert "system" in categories
        assert "calculator" in categories["utility"]
        assert "file_ops" in categories["system"]
        
        # Empty categories should not be included
        assert "development" not in categories
    
    def test_get_required_permissions(self):
        """Test getting required permissions."""
        registry = ToolRegistry()
        calc_tool = MockCalculatorTool()
        file_tool = MockFileOpsTool()
        
        registry.register(calc_tool)
        registry.register(file_tool)
        
        permissions = registry.get_required_permissions()
        
        assert ToolPermission.READ in permissions
        assert ToolPermission.WRITE in permissions
        assert len(permissions) == 2
        
        # Should return a copy
        permissions.add(ToolPermission.EXECUTE)
        assert ToolPermission.EXECUTE not in registry._permissions_required
    
    @pytest.mark.asyncio
    async def test_execute_tool(self):
        """Test executing tool through registry."""
        registry = ToolRegistry()
        calc_tool = MockCalculatorTool()
        registry.register(calc_tool)
        
        # Execute existing tool
        result = await registry.execute_tool("calculator", expression="2+2")
        
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.output == 4
        
        # Execute non-existing tool
        result = await registry.execute_tool("nonexistent")
        
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert "not found" in result.error
        assert "available_tools" in result.metadata
    
    def test_get_tool_info(self):
        """Test getting tool information."""
        registry = ToolRegistry()
        calc_tool = MockCalculatorTool()
        registry.register(calc_tool)
        
        # Get info for existing tool
        info = registry.get_tool_info("calculator")
        
        assert info is not None
        assert info["name"] == "calculator"
        assert info["description"] == "Basic calculator operations"
        assert info["category"] == "utility"
        assert info["permissions"] == ["read"]
        assert len(info["parameters"]) == 1
        assert info["parameters"][0]["name"] == "expression"
        
        # Get info for non-existing tool
        info = registry.get_tool_info("nonexistent")
        assert info is None
    
    def test_clear_registry(self):
        """Test clearing the registry."""
        registry = ToolRegistry()
        calc_tool = MockCalculatorTool()
        file_tool = MockFileOpsTool()
        
        registry.register(calc_tool)
        registry.register(file_tool)
        
        assert len(registry._tools) == 2
        assert len(registry._permissions_required) > 0
        
        registry.clear()
        
        assert len(registry._tools) == 0
        assert len(registry._permissions_required) == 0
        for category in ToolCategory:
            assert len(registry._categories[category]) == 0


@pytest.mark.unit
class TestToolDiscovery:
    """Test ToolDiscovery class."""
    
    def test_discovery_initialization(self):
        """Test discovery initialization."""
        registry = ToolRegistry()
        discovery = ToolDiscovery(registry)
        
        assert discovery.registry is registry
    
    @patch('transcriber.tools.registry.importlib.import_module')
    @patch('transcriber.tools.registry.inspect.getmembers')
    def test_discover_in_module(self, mock_getmembers, mock_import):
        """Test discovering tools in a module."""
        registry = ToolRegistry()
        discovery = ToolDiscovery(registry)
        
        # Mock module with tool classes
        mock_module = MagicMock()
        mock_import.return_value = mock_module
        
        # Mock inspect.getmembers to return tool classes
        mock_getmembers.return_value = [
            ("MockCalculatorTool", MockCalculatorTool),
            ("NotATool", str),  # Should be ignored
            ("BaseTool", BaseTool),  # Should be ignored (abstract)
        ]
        
        discovered = discovery.discover_in_module("test.module")
        
        assert len(discovered) == 1
        assert "calculator" in discovered
        assert "calculator" in registry._tools
    
    @patch('transcriber.tools.registry.importlib.import_module')
    def test_discover_in_module_import_error(self, mock_import):
        """Test discovery with import error."""
        registry = ToolRegistry()
        discovery = ToolDiscovery(registry)
        
        mock_import.side_effect = ImportError("Module not found")
        
        discovered = discovery.discover_in_module("nonexistent.module")
        
        assert len(discovered) == 0
    
    @patch('transcriber.tools.registry.importlib.import_module')
    @patch('transcriber.tools.registry.inspect.getmembers')
    def test_discover_in_module_instantiation_error(self, mock_getmembers, mock_import):
        """Test discovery with tool instantiation error."""
        registry = ToolRegistry()
        discovery = ToolDiscovery(registry)
        
        # Mock module
        mock_module = MagicMock()
        mock_import.return_value = mock_module
        
        # Create a tool class that raises error on instantiation
        class BrokenTool(BaseTool):
            def __init__(self):
                raise ValueError("Broken tool")
            
            def _define_metadata(self):
                return ToolMetadata(
                    name="broken",
                    description="Broken tool",
                    category=ToolCategory.UTILITY
                )
            
            def _define_parameters(self):
                return []
            
            async def _execute(self, **kwargs):
                return "test"
        
        mock_getmembers.return_value = [("BrokenTool", BrokenTool)]
        
        discovered = discovery.discover_in_module("test.module")
        
        assert len(discovered) == 0  # Should handle error gracefully
    
    def test_discover_in_directory(self):
        """Test discovering tools in a directory."""
        registry = ToolRegistry()
        discovery = ToolDiscovery(registry)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create some Python files
            (temp_path / "tool1.py").write_text("# Tool 1")
            (temp_path / "tool2.py").write_text("# Tool 2")
            (temp_path / "_private.py").write_text("# Private")  # Should be ignored
            (temp_path / "not_python.txt").write_text("Not Python")  # Should be ignored
            
            # Mock the module discovery
            with patch.object(discovery, 'discover_in_module') as mock_discover:
                mock_discover.return_value = ["mock_tool"]
                
                discovered = discovery.discover_in_directory(temp_path)
                
                # Should call discover_in_module for each Python file (except private)
                assert mock_discover.call_count == 2
                assert len(discovered) == 2  # 2 calls * 1 tool each
    
    def test_discover_in_nonexistent_directory(self):
        """Test discovery in non-existent directory."""
        registry = ToolRegistry()
        discovery = ToolDiscovery(registry)
        
        discovered = discovery.discover_in_directory("/nonexistent/path")
        
        assert len(discovered) == 0
    
    @patch.object(Path, 'exists')
    @patch.object(ToolDiscovery, 'discover_in_directory')
    def test_auto_discover(self, mock_discover_dir, mock_exists):
        """Test automatic tool discovery."""
        registry = ToolRegistry()
        discovery = ToolDiscovery(registry)
        
        # Mock directory existence
        mock_exists.return_value = True
        mock_discover_dir.return_value = ["tool1", "tool2"]
        
        discovered = discovery.auto_discover()
        
        # Should discover in builtin directory
        assert mock_discover_dir.call_count >= 1
        assert len(discovered) >= 2
    
    @patch.object(Path, 'exists')
    def test_auto_discover_no_directories(self, mock_exists):
        """Test auto discovery when no directories exist."""
        registry = ToolRegistry()
        discovery = ToolDiscovery(registry)
        
        mock_exists.return_value = False
        
        discovered = discovery.auto_discover()
        
        assert len(discovered) == 0


@pytest.mark.unit
class TestToolRegistryIntegration:
    """Test tool registry integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_tool_lifecycle(self):
        """Test complete tool lifecycle in registry."""
        registry = ToolRegistry()
        
        # Register multiple tools
        calc_tool = MockCalculatorTool()
        file_tool = MockFileOpsTool()
        
        registry.register(calc_tool)
        registry.register(file_tool)
        
        # Verify registration
        assert len(registry.list_tools()) == 2
        
        # Execute tools
        calc_result = await registry.execute_tool("calculator", expression="2+2")
        assert calc_result.success is True
        assert calc_result.output == 4
        
        file_result = await registry.execute_tool(
            "file_ops",
            operation="read",
            path="/test/path"
        )
        assert file_result.success is True
        assert "read" in file_result.output
        
        # Search and filter
        utility_tools = registry.get_by_category(ToolCategory.UTILITY)
        assert len(utility_tools) == 1
        
        read_tools = registry.get_by_permission(ToolPermission.READ)
        assert len(read_tools) == 2
        
        # Unregister tool
        registry.unregister("calculator")
        assert len(registry.list_tools()) == 1
        
        # Clear registry
        registry.clear()
        assert len(registry.list_tools()) == 0
    
    def test_registry_with_complex_tool_metadata(self):
        """Test registry with complex tool metadata."""
        
        class ComplexTool(BaseTool):
            def _define_metadata(self):
                return ToolMetadata(
                    name="complex_tool",
                    description="A complex tool with many features",
                    category=ToolCategory.DEVELOPMENT,
                    version="2.1.0",
                    author="Test Author",
                    permissions=[
                        ToolPermission.READ,
                        ToolPermission.WRITE,
                        ToolPermission.EXECUTE
                    ],
                    examples=[
                        "complex_tool(mode='fast', count=10)",
                        "complex_tool(mode='slow', count=1, debug=True)"
                    ]
                )
            
            def _define_parameters(self):
                return [
                    ToolParameter(
                        name="mode",
                        type="str",
                        description="Processing mode",
                        required=True,
                        choices=["fast", "slow"]
                    ),
                    ToolParameter(
                        name="count",
                        type="int",
                        description="Number of iterations",
                        required=False,
                        default=1
                    ),
                    ToolParameter(
                        name="debug",
                        type="bool",
                        description="Enable debug mode",
                        required=False,
                        default=False
                    )
                ]
            
            async def _execute(self, **kwargs):
                return f"Complex processing: {kwargs}"
        
        registry = ToolRegistry()
        tool = ComplexTool()
        registry.register(tool)
        
        # Test detailed tool info
        info = registry.get_tool_info("complex_tool")
        assert info["version"] == "2.1.0"
        assert info["author"] == "Test Author"
        assert len(info["permissions"]) == 3
        assert len(info["examples"]) == 2
        assert len(info["parameters"]) == 3
        
        # Test permission filtering
        exec_tools = registry.get_by_permission(ToolPermission.EXECUTE)
        assert len(exec_tools) == 1
        assert exec_tools[0] is tool
        
        # Test category filtering
        dev_tools = registry.get_by_category(ToolCategory.DEVELOPMENT)
        assert len(dev_tools) == 1
        assert dev_tools[0] is tool