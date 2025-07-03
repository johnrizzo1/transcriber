"""
File operations tools for reading, writing, and managing files.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..base import BaseTool, ToolCategory, ToolMetadata, ToolParameter, ToolPermission


class FileReadTool(BaseTool):
    """Tool for reading file contents."""
    
    def _define_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="file_read",
            description="Read the contents of a file",
            category=ToolCategory.SYSTEM,
            version="1.0.0",
            author="System",
            permissions=[ToolPermission.READ],
            examples=[
                'file_read(path="/path/to/file.txt")',
                'file_read(path="~/documents/notes.md", encoding="utf-8")',
                'file_read(path="config.json", lines=10)'
            ]
        )
    
    def _define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="str",
                description="Path to the file to read",
                required=True
            ),
            ToolParameter(
                name="encoding",
                type="str",
                description="File encoding (default: utf-8)",
                required=False,
                default="utf-8"
            ),
            ToolParameter(
                name="lines",
                type="int",
                description="Number of lines to read (default: all)",
                required=False,
                default=None
            )
        ]
    
    async def _execute(self, path: str, encoding: str = "utf-8", lines: Optional[int] = None) -> str:
        """Read file contents."""
        file_path = Path(path).expanduser().resolve()
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                if lines:
                    content = ''.join(f.readline() for _ in range(lines))
                else:
                    content = f.read()
            
            return content
            
        except Exception as e:
            raise RuntimeError(f"Failed to read file: {e}")


class FileWriteTool(BaseTool):
    """Tool for writing content to files."""
    
    def _define_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="file_write",
            description="Write content to a file",
            category=ToolCategory.SYSTEM,
            version="1.0.0",
            author="System",
            permissions=[ToolPermission.WRITE],
            examples=[
                'file_write(path="/path/to/file.txt", content="Hello World")',
                'file_write(path="output.log", content="Log entry", append=true)',
                'file_write(path="~/notes.md", content="# Notes\\n\\nContent here")'
            ]
        )
    
    def _define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="str",
                description="Path to the file to write",
                required=True
            ),
            ToolParameter(
                name="content",
                type="str",
                description="Content to write to the file",
                required=True
            ),
            ToolParameter(
                name="append",
                type="bool",
                description="Append to file instead of overwriting",
                required=False,
                default=False
            ),
            ToolParameter(
                name="encoding",
                type="str",
                description="File encoding (default: utf-8)",
                required=False,
                default="utf-8"
            )
        ]
    
    async def _execute(
        self, 
        path: str, 
        content: str, 
        append: bool = False, 
        encoding: str = "utf-8"
    ) -> Dict[str, Any]:
        """Write content to file."""
        file_path = Path(path).expanduser().resolve()
        
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        mode = 'a' if append else 'w'
        
        try:
            with open(file_path, mode, encoding=encoding) as f:
                f.write(content)
            
            return {
                "path": str(file_path),
                "size": len(content),
                "mode": "append" if append else "write",
                "success": True
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to write file: {e}")


class FileListTool(BaseTool):
    """Tool for listing files in a directory."""
    
    def _define_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="file_list",
            description="List files and directories in a given path",
            category=ToolCategory.SYSTEM,
            version="1.0.0",
            author="System",
            permissions=[ToolPermission.READ],
            examples=[
                'file_list(path="/home/user/documents")',
                'file_list(path=".", pattern="*.py")',
                'file_list(path="~/projects", recursive=true, pattern="*.md")'
            ]
        )
    
    def _define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="str",
                description="Path to the directory to list",
                required=True
            ),
            ToolParameter(
                name="pattern",
                type="str",
                description="File pattern to match (e.g., *.txt)",
                required=False,
                default=None
            ),
            ToolParameter(
                name="recursive",
                type="bool",
                description="List files recursively",
                required=False,
                default=False
            ),
            ToolParameter(
                name="include_hidden",
                type="bool",
                description="Include hidden files (starting with .)",
                required=False,
                default=False
            )
        ]
    
    async def _execute(
        self, 
        path: str, 
        pattern: Optional[str] = None,
        recursive: bool = False,
        include_hidden: bool = False
    ) -> List[Dict[str, Any]]:
        """List files in directory."""
        dir_path = Path(path).expanduser().resolve()
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {dir_path}")
        
        files = []
        
        try:
            if recursive and pattern:
                # Use glob for recursive pattern matching
                for item in dir_path.rglob(pattern):
                    if not include_hidden and item.name.startswith('.'):
                        continue
                    files.append(self._get_file_info(item))
            elif recursive:
                # Walk through all subdirectories
                for root, dirs, filenames in os.walk(dir_path):
                    # Filter hidden directories
                    if not include_hidden:
                        dirs[:] = [d for d in dirs if not d.startswith('.')]
                    
                    root_path = Path(root)
                    for name in filenames:
                        if not include_hidden and name.startswith('.'):
                            continue
                        files.append(self._get_file_info(root_path / name))
            else:
                # List immediate directory contents
                pattern_obj = pattern or "*"
                for item in dir_path.glob(pattern_obj):
                    if not include_hidden and item.name.startswith('.'):
                        continue
                    files.append(self._get_file_info(item))
            
            # Sort by path
            files.sort(key=lambda x: x['path'])
            
            return files
            
        except Exception as e:
            raise RuntimeError(f"Failed to list directory: {e}")
    
    def _get_file_info(self, path: Path) -> Dict[str, Any]:
        """Get information about a file or directory."""
        stat = path.stat()
        
        return {
            "path": str(path),
            "name": path.name,
            "type": "directory" if path.is_dir() else "file",
            "size": stat.st_size if path.is_file() else None,
            "modified": stat.st_mtime,
            "permissions": oct(stat.st_mode)[-3:]
        }


class FileDeleteTool(BaseTool):
    """Tool for deleting files and directories."""
    
    def _define_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="file_delete",
            description="Delete a file or directory",
            category=ToolCategory.SYSTEM,
            version="1.0.0",
            author="System",
            permissions=[ToolPermission.WRITE],
            examples=[
                'file_delete(path="/tmp/temp_file.txt")',
                'file_delete(path="~/old_directory", recursive=true)',
                'file_delete(path="backup.zip", confirm=true)'
            ]
        )
    
    def _define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="str",
                description="Path to the file or directory to delete",
                required=True
            ),
            ToolParameter(
                name="recursive",
                type="bool",
                description="Delete directories recursively",
                required=False,
                default=False
            ),
            ToolParameter(
                name="confirm",
                type="bool",
                description="Confirmation flag for safety",
                required=False,
                default=False
            )
        ]
    
    async def _execute(
        self, 
        path: str, 
        recursive: bool = False,
        confirm: bool = False
    ) -> Dict[str, Any]:
        """Delete file or directory."""
        if not confirm:
            raise ValueError("Deletion requires confirm=true for safety")
        
        file_path = Path(path).expanduser().resolve()
        
        if not file_path.exists():
            raise FileNotFoundError(f"Path not found: {file_path}")
        
        try:
            if file_path.is_file():
                file_path.unlink()
                return {
                    "path": str(file_path),
                    "type": "file",
                    "deleted": True
                }
            elif file_path.is_dir():
                if recursive:
                    shutil.rmtree(file_path)
                else:
                    file_path.rmdir()  # Only works on empty directories
                
                return {
                    "path": str(file_path),
                    "type": "directory",
                    "deleted": True,
                    "recursive": recursive
                }
            
        except Exception as e:
            raise RuntimeError(f"Failed to delete: {e}")


class FileCopyTool(BaseTool):
    """Tool for copying files and directories."""
    
    def _define_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="file_copy",
            description="Copy a file or directory to a new location",
            category=ToolCategory.SYSTEM,
            version="1.0.0",
            author="System",
            permissions=[ToolPermission.READ, ToolPermission.WRITE],
            examples=[
                'file_copy(source="/path/to/file.txt", destination="/backup/file.txt")',
                'file_copy(source="~/documents", destination="~/backup/documents", recursive=true)',
            ]
        )
    
    def _define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="source",
                type="str",
                description="Source path to copy from",
                required=True
            ),
            ToolParameter(
                name="destination",
                type="str",
                description="Destination path to copy to",
                required=True
            ),
            ToolParameter(
                name="recursive",
                type="bool",
                description="Copy directories recursively",
                required=False,
                default=False
            ),
            ToolParameter(
                name="overwrite",
                type="bool",
                description="Overwrite existing files",
                required=False,
                default=False
            )
        ]
    
    async def _execute(
        self, 
        source: str, 
        destination: str,
        recursive: bool = False,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """Copy file or directory."""
        src_path = Path(source).expanduser().resolve()
        dst_path = Path(destination).expanduser().resolve()
        
        if not src_path.exists():
            raise FileNotFoundError(f"Source not found: {src_path}")
        
        if dst_path.exists() and not overwrite:
            raise FileExistsError(f"Destination already exists: {dst_path}")
        
        try:
            if src_path.is_file():
                # Create parent directories
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
                
                return {
                    "source": str(src_path),
                    "destination": str(dst_path),
                    "type": "file",
                    "size": src_path.stat().st_size
                }
                
            elif src_path.is_dir():
                if not recursive:
                    raise ValueError("Directory copy requires recursive=true")
                
                if dst_path.exists():
                    shutil.rmtree(dst_path)
                
                shutil.copytree(src_path, dst_path)
                
                # Count files copied
                file_count = sum(1 for _ in dst_path.rglob("*") if _.is_file())
                
                return {
                    "source": str(src_path),
                    "destination": str(dst_path),
                    "type": "directory",
                    "files_copied": file_count
                }
                
        except Exception as e:
            raise RuntimeError(f"Failed to copy: {e}")