"""
Unit tests for session management models.
"""

import json
from datetime import datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from transcriber.session.models import (
    SessionStatus,
    MessageType,
    SessionMessage,
    SessionMetadata,
    Session
)


@pytest.mark.unit
class TestSessionEnums:
    """Test session enumeration classes."""
    
    def test_session_status_values(self):
        """Test SessionStatus enum values."""
        assert SessionStatus.ACTIVE.value == "active"
        assert SessionStatus.COMPLETED.value == "completed"
        assert SessionStatus.ARCHIVED.value == "archived"
    
    def test_message_type_values(self):
        """Test MessageType enum values."""
        assert MessageType.USER.value == "user"
        assert MessageType.ASSISTANT.value == "assistant"
        assert MessageType.SYSTEM.value == "system"
        assert MessageType.TOOL.value == "tool"


@pytest.mark.unit
class TestSessionMessage:
    """Test SessionMessage model."""
    
    def test_basic_message_creation(self):
        """Test basic message creation."""
        message = SessionMessage(
            content="Hello, world!",
            type=MessageType.USER
        )
        
        assert message.content == "Hello, world!"
        assert message.type == MessageType.USER
        assert isinstance(message.id, UUID)
        assert isinstance(message.timestamp, datetime)
        assert message.metadata == {}
    
    def test_message_with_custom_id(self):
        """Test message with custom ID."""
        custom_id = uuid4()
        message = SessionMessage(
            id=custom_id,
            content="Test message",
            type=MessageType.ASSISTANT
        )
        
        assert message.id == custom_id
    
    def test_message_with_custom_timestamp(self):
        """Test message with custom timestamp."""
        custom_time = datetime(2024, 1, 1, 12, 0, 0)
        message = SessionMessage(
            content="Test message",
            type=MessageType.SYSTEM,
            timestamp=custom_time
        )
        
        assert message.timestamp == custom_time
    
    def test_message_with_metadata(self):
        """Test message with metadata."""
        metadata = {
            "tool_name": "calculator",
            "execution_time": 0.5,
            "success": True
        }
        message = SessionMessage(
            content="Calculation result: 42",
            type=MessageType.TOOL,
            metadata=metadata
        )
        
        assert message.metadata == metadata
    
    def test_message_validation(self):
        """Test message validation."""
        # Valid message should work
        SessionMessage(content="Test", type=MessageType.USER)
        
        # Missing required fields should raise ValidationError
        with pytest.raises(ValidationError):
            SessionMessage(content="Test")  # Missing type
        
        with pytest.raises(ValidationError):
            SessionMessage(type=MessageType.USER)  # Missing content
    
    def test_message_serialization(self):
        """Test message serialization."""
        message = SessionMessage(
            content="Test message",
            type=MessageType.USER,
            metadata={"key": "value"}
        )
        
        # Should be able to convert to dict
        message_dict = message.model_dump()
        assert isinstance(message_dict, dict)
        assert message_dict["content"] == "Test message"
        assert message_dict["type"] == "user"
        assert message_dict["metadata"] == {"key": "value"}
        
        # Should be able to convert to JSON
        message_json = message.model_dump_json()
        assert isinstance(message_json, str)
        
        # Should be able to parse back from dict
        new_message = SessionMessage(**message_dict)
        assert new_message.content == message.content
        assert new_message.type == message.type
        assert new_message.metadata == message.metadata
    
    def test_message_string_representation(self):
        """Test message string representation."""
        message = SessionMessage(
            content="Hello, world!",
            type=MessageType.USER
        )
        
        str_repr = str(message)
        assert "Hello, world!" in str_repr
        assert "user" in str_repr
    
    def test_message_equality(self):
        """Test message equality comparison."""
        message_id = uuid4()
        timestamp = datetime.now()
        
        message1 = SessionMessage(
            id=message_id,
            content="Test",
            type=MessageType.USER,
            timestamp=timestamp
        )
        
        message2 = SessionMessage(
            id=message_id,
            content="Test",
            type=MessageType.USER,
            timestamp=timestamp
        )
        
        # Messages with same ID should be equal
        assert message1 == message2
        
        # Messages with different content but same ID should still be equal
        message3 = SessionMessage(
            id=message_id,
            content="Different content",
            type=MessageType.ASSISTANT,
            timestamp=timestamp
        )
        
        assert message1 == message3  # Based on ID


@pytest.mark.unit
class TestSessionMetadata:
    """Test SessionMetadata model."""
    
    def test_basic_metadata_creation(self):
        """Test basic metadata creation."""
        metadata = SessionMetadata()
        
        assert metadata.tags == []
        assert metadata.user_id is None
        assert metadata.session_type is None
        assert metadata.custom_data == {}
    
    def test_metadata_with_values(self):
        """Test metadata with custom values."""
        tags = ["important", "work", "project-x"]
        custom_data = {
            "project": "AI Assistant",
            "version": "1.0",
            "priority": "high"
        }
        
        metadata = SessionMetadata(
            tags=tags,
            user_id="user123",
            session_type="voice_chat",
            custom_data=custom_data
        )
        
        assert metadata.tags == tags
        assert metadata.user_id == "user123"
        assert metadata.session_type == "voice_chat"
        assert metadata.custom_data == custom_data
    
    def test_metadata_serialization(self):
        """Test metadata serialization."""
        metadata = SessionMetadata(
            tags=["test"],
            user_id="user123",
            custom_data={"key": "value"}
        )
        
        # Should serialize to dict
        metadata_dict = metadata.model_dump()
        assert isinstance(metadata_dict, dict)
        assert metadata_dict["tags"] == ["test"]
        assert metadata_dict["user_id"] == "user123"
        
        # Should serialize to JSON
        metadata_json = metadata.model_dump_json()
        assert isinstance(metadata_json, str)
        
        # Should parse back from dict
        new_metadata = SessionMetadata(**metadata_dict)
        assert new_metadata.tags == metadata.tags
        assert new_metadata.user_id == metadata.user_id


@pytest.mark.unit
class TestSession:
    """Test Session model."""
    
    def test_basic_session_creation(self):
        """Test basic session creation."""
        session = Session(title="Test Session")
        
        assert session.title == "Test Session"
        assert isinstance(session.id, UUID)
        assert session.status == SessionStatus.ACTIVE
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.updated_at, datetime)
        assert session.completed_at is None
        assert len(session.messages) == 0
        assert isinstance(session.metadata, SessionMetadata)
    
    def test_session_with_custom_values(self):
        """Test session with custom values."""
        session_id = uuid4()
        created_time = datetime(2024, 1, 1, 10, 0, 0)
        updated_time = datetime(2024, 1, 1, 11, 0, 0)
        completed_time = datetime(2024, 1, 1, 12, 0, 0)
        
        metadata = SessionMetadata(
            tags=["test"],
            user_id="user123"
        )
        
        session = Session(
            id=session_id,
            title="Custom Session",
            status=SessionStatus.COMPLETED,
            created_at=created_time,
            updated_at=updated_time,
            completed_at=completed_time,
            metadata=metadata
        )
        
        assert session.id == session_id
        assert session.status == SessionStatus.COMPLETED
        assert session.created_at == created_time
        assert session.updated_at == updated_time
        assert session.completed_at == completed_time
        assert session.metadata == metadata
    
    def test_session_with_messages(self):
        """Test session with messages."""
        messages = [
            SessionMessage(content="Hello", type=MessageType.USER),
            SessionMessage(content="Hi there!", type=MessageType.ASSISTANT)
        ]
        
        session = Session(
            title="Chat Session",
            messages=messages
        )
        
        assert len(session.messages) == 2
        assert session.messages[0].content == "Hello"
        assert session.messages[1].content == "Hi there!"
    
    def test_session_validation(self):
        """Test session validation."""
        # Valid session should work
        Session(title="Valid Session")
        
        # Missing title should raise ValidationError
        with pytest.raises(ValidationError):
            Session()
        
        # Empty title should raise ValidationError
        with pytest.raises(ValidationError):
            Session(title="")
    
    def test_add_message(self):
        """Test adding messages to session."""
        session = Session(title="Test Session")
        
        message1 = SessionMessage(content="First", type=MessageType.USER)
        message2 = SessionMessage(content="Second", type=MessageType.ASSISTANT)
        
        session.add_message(message1)
        assert len(session.messages) == 1
        assert session.messages[0] == message1
        
        session.add_message(message2)
        assert len(session.messages) == 2
        assert session.messages[1] == message2
    
    def test_get_message_by_id(self):
        """Test getting message by ID."""
        session = Session(title="Test Session")
        
        message = SessionMessage(content="Test", type=MessageType.USER)
        session.add_message(message)
        
        # Get existing message
        retrieved = session.get_message(message.id)
        assert retrieved == message
        
        # Get non-existing message
        non_existing_id = uuid4()
        retrieved = session.get_message(non_existing_id)
        assert retrieved is None
    
    def test_get_messages_by_type(self):
        """Test getting messages by type."""
        session = Session(title="Test Session")
        
        user_msg1 = SessionMessage(content="User 1", type=MessageType.USER)
        assistant_msg = SessionMessage(content="Assistant", type=MessageType.ASSISTANT)
        user_msg2 = SessionMessage(content="User 2", type=MessageType.USER)
        
        session.add_message(user_msg1)
        session.add_message(assistant_msg)
        session.add_message(user_msg2)
        
        # Get user messages
        user_messages = session.get_messages_by_type(MessageType.USER)
        assert len(user_messages) == 2
        assert user_msg1 in user_messages
        assert user_msg2 in user_messages
        
        # Get assistant messages
        assistant_messages = session.get_messages_by_type(MessageType.ASSISTANT)
        assert len(assistant_messages) == 1
        assert assistant_msg in assistant_messages
        
        # Get non-existing type
        system_messages = session.get_messages_by_type(MessageType.SYSTEM)
        assert len(system_messages) == 0
    
    def test_session_duration(self):
        """Test session duration calculation."""
        created_time = datetime(2024, 1, 1, 10, 0, 0)
        completed_time = datetime(2024, 1, 1, 10, 30, 0)  # 30 minutes later
        
        # Active session (no completion time)
        active_session = Session(
            title="Active Session",
            created_at=created_time
        )
        duration = active_session.get_duration()
        assert duration is not None  # Should return current duration
        
        # Completed session
        completed_session = Session(
            title="Completed Session",
            status=SessionStatus.COMPLETED,
            created_at=created_time,
            completed_at=completed_time
        )
        duration = completed_session.get_duration()
        assert duration.total_seconds() == 1800  # 30 minutes
    
    def test_session_serialization(self):
        """Test session serialization."""
        messages = [
            SessionMessage(content="Hello", type=MessageType.USER),
            SessionMessage(content="Hi!", type=MessageType.ASSISTANT)
        ]
        
        metadata = SessionMetadata(
            tags=["test"],
            user_id="user123"
        )
        
        session = Session(
            title="Test Session",
            messages=messages,
            metadata=metadata
        )
        
        # Should serialize to dict
        session_dict = session.model_dump()
        assert isinstance(session_dict, dict)
        assert session_dict["title"] == "Test Session"
        assert len(session_dict["messages"]) == 2
        assert session_dict["metadata"]["tags"] == ["test"]
        
        # Should serialize to JSON
        session_json = session.model_dump_json()
        assert isinstance(session_json, str)
        
        # Should parse back from dict
        new_session = Session(**session_dict)
        assert new_session.title == session.title
        assert len(new_session.messages) == len(session.messages)
        assert new_session.metadata.tags == session.metadata.tags
    
    def test_session_export_format(self):
        """Test session export format."""
        messages = [
            SessionMessage(
                content="What's 2+2?",
                type=MessageType.USER,
                timestamp=datetime(2024, 1, 1, 10, 0, 0)
            ),
            SessionMessage(
                content="2+2 equals 4",
                type=MessageType.ASSISTANT,
                timestamp=datetime(2024, 1, 1, 10, 0, 1)
            )
        ]
        
        session = Session(
            title="Math Session",
            messages=messages,
            created_at=datetime(2024, 1, 1, 10, 0, 0),
            completed_at=datetime(2024, 1, 1, 10, 5, 0)
        )
        
        # Test export to dict format
        export_data = session.to_export_format()
        
        assert export_data["title"] == "Math Session"
        assert export_data["status"] == "active"
        assert len(export_data["messages"]) == 2
        assert export_data["message_count"] == 2
        assert export_data["duration_seconds"] == 300  # 5 minutes
        
        # Check message format
        msg = export_data["messages"][0]
        assert msg["content"] == "What's 2+2?"
        assert msg["type"] == "user"
        assert "timestamp" in msg
    
    def test_session_statistics(self):
        """Test session statistics calculation."""
        messages = [
            SessionMessage(content="User 1", type=MessageType.USER),
            SessionMessage(content="Assistant 1", type=MessageType.ASSISTANT),
            SessionMessage(content="User 2", type=MessageType.USER),
            SessionMessage(content="Tool result", type=MessageType.TOOL),
            SessionMessage(content="Assistant 2", type=MessageType.ASSISTANT)
        ]
        
        session = Session(title="Stats Session", messages=messages)
        stats = session.get_statistics()
        
        assert stats["total_messages"] == 5
        assert stats["user_messages"] == 2
        assert stats["assistant_messages"] == 2
        assert stats["tool_messages"] == 1
        assert stats["system_messages"] == 0
        
        # Check message type distribution
        assert stats["message_types"]["user"] == 2
        assert stats["message_types"]["assistant"] == 2
        assert stats["message_types"]["tool"] == 1
        assert stats["message_types"]["system"] == 0


@pytest.mark.unit
class TestSessionModelIntegration:
    """Test session model integration scenarios."""
    
    def test_complete_session_workflow(self):
        """Test complete session workflow."""
        # Create new session
        session = Session(title="Integration Test Session")
        assert session.status == SessionStatus.ACTIVE
        
        # Add user message
        user_msg = SessionMessage(
            content="Hello, can you help me?",
            type=MessageType.USER
        )
        session.add_message(user_msg)
        
        # Add assistant response
        assistant_msg = SessionMessage(
            content="Of course! How can I help you?",
            type=MessageType.ASSISTANT
        )
        session.add_message(assistant_msg)
        
        # Add tool execution
        tool_msg = SessionMessage(
            content="Executed calculator tool",
            type=MessageType.TOOL,
            metadata={
                "tool_name": "calculator",
                "input": "2+2",
                "output": "4"
            }
        )
        session.add_message(tool_msg)
        
        # Complete session
        session.status = SessionStatus.COMPLETED
        session.completed_at = datetime.now()
        
        # Verify final state
        assert len(session.messages) == 3
        assert session.status == SessionStatus.COMPLETED
        assert session.completed_at is not None
        
        # Check statistics
        stats = session.get_statistics()
        assert stats["total_messages"] == 3
        assert stats["user_messages"] == 1
        assert stats["assistant_messages"] == 1
        assert stats["tool_messages"] == 1
        
        # Test serialization
        session_dict = session.model_dump()
        restored_session = Session(**session_dict)
        assert restored_session.title == session.title
        assert len(restored_session.messages) == len(session.messages)
    
    def test_session_with_complex_metadata(self):
        """Test session with complex metadata."""
        metadata = SessionMetadata(
            tags=["important", "work", "ai-assistant"],
            user_id="user_12345",
            session_type="voice_conversation",
            custom_data={
                "project": "AI Voice Agent",
                "version": "1.0.0",
                "environment": "production",
                "features_used": ["voice", "tools", "memory"],
                "performance_metrics": {
                    "avg_response_time": 0.8,
                    "total_tokens": 1500,
                    "audio_quality": "high"
                }
            }
        )
        
        session = Session(
            title="Complex Metadata Session",
            metadata=metadata
        )
        
        # Verify metadata is preserved
        assert session.metadata.tags == ["important", "work", "ai-assistant"]
        assert session.metadata.user_id == "user_12345"
        assert session.metadata.session_type == "voice_conversation"
        assert "project" in session.metadata.custom_data
        assert "performance_metrics" in session.metadata.custom_data
        
        # Test serialization preserves complex metadata
        session_dict = session.model_dump()
        restored_session = Session(**session_dict)
        
        assert restored_session.metadata.tags == metadata.tags
        assert restored_session.metadata.custom_data == metadata.custom_data