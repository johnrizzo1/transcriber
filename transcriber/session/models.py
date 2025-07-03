"""
Data models for session management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class SessionStatus(Enum):
    """Session status enumeration."""
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class MessageType(Enum):
    """Message type enumeration."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class SessionMessage:
    """Individual message within a session."""
    id: UUID = field(default_factory=uuid4)
    session_id: UUID = field(default_factory=uuid4)
    content: str = ""
    message_type: MessageType = MessageType.USER
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "id": str(self.id),
            "session_id": str(self.session_id),
            "content": self.content,
            "message_type": self.message_type.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionMessage":
        """Create message from dictionary."""
        return cls(
            id=UUID(data["id"]),
            session_id=UUID(data["session_id"]),
            content=data["content"],
            message_type=MessageType(data["message_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata")
        )


@dataclass
class SessionMetadata:
    """Session metadata and statistics."""
    total_messages: int = 0
    user_messages: int = 0
    assistant_messages: int = 0
    tool_executions: int = 0
    duration_seconds: Optional[float] = None
    audio_recorded: bool = False
    audio_file_path: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    custom_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "total_messages": self.total_messages,
            "user_messages": self.user_messages,
            "assistant_messages": self.assistant_messages,
            "tool_executions": self.tool_executions,
            "duration_seconds": self.duration_seconds,
            "audio_recorded": self.audio_recorded,
            "audio_file_path": self.audio_file_path,
            "tags": self.tags,
            "custom_data": self.custom_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionMetadata":
        """Create metadata from dictionary."""
        return cls(
            total_messages=data.get("total_messages", 0),
            user_messages=data.get("user_messages", 0),
            assistant_messages=data.get("assistant_messages", 0),
            tool_executions=data.get("tool_executions", 0),
            duration_seconds=data.get("duration_seconds"),
            audio_recorded=data.get("audio_recorded", False),
            audio_file_path=data.get("audio_file_path"),
            tags=data.get("tags", []),
            custom_data=data.get("custom_data", {})
        )


@dataclass
class Session:
    """Complete session with messages and metadata."""
    id: UUID = field(default_factory=uuid4)
    title: str = ""
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    messages: List[SessionMessage] = field(default_factory=list)
    metadata: SessionMetadata = field(default_factory=SessionMetadata)
    
    def add_message(self, message: SessionMessage) -> None:
        """Add a message to the session."""
        message.session_id = self.id
        self.messages.append(message)
        self.updated_at = datetime.now()
        
        # Update metadata
        self.metadata.total_messages += 1
        if message.message_type == MessageType.USER:
            self.metadata.user_messages += 1
        elif message.message_type == MessageType.ASSISTANT:
            self.metadata.assistant_messages += 1
        elif message.message_type == MessageType.TOOL:
            self.metadata.tool_executions += 1
    
    def complete(self) -> None:
        """Mark session as completed."""
        self.status = SessionStatus.COMPLETED
        self.completed_at = datetime.now()
        self.updated_at = self.completed_at
        
        # Calculate duration
        if self.created_at and self.completed_at:
            duration = self.completed_at - self.created_at
            self.metadata.duration_seconds = duration.total_seconds()
    
    def get_conversation_text(self) -> str:
        """Get formatted conversation text."""
        lines = []
        for msg in self.messages:
            timestamp = msg.timestamp.strftime("%H:%M:%S")
            if msg.message_type == MessageType.USER:
                lines.append(f"[{timestamp}] User: {msg.content}")
            elif msg.message_type == MessageType.ASSISTANT:
                lines.append(f"[{timestamp}] Assistant: {msg.content}")
            elif msg.message_type == MessageType.TOOL:
                lines.append(f"[{timestamp}] Tool: {msg.content}")
            elif msg.message_type == MessageType.SYSTEM:
                lines.append(f"[{timestamp}] System: {msg.content}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "id": str(self.id),
            "title": self.title,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "messages": [msg.to_dict() for msg in self.messages],
            "metadata": self.metadata.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Create session from dictionary."""
        session = cls(
            id=UUID(data["id"]),
            title=data["title"],
            status=SessionStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            metadata=SessionMetadata.from_dict(data.get("metadata", {}))
        )
        
        # Load messages
        for msg_data in data.get("messages", []):
            message = SessionMessage.from_dict(msg_data)
            session.messages.append(message)
        
        return session