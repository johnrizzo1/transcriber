"""
Session manager for coordinating session operations.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import UUID, uuid4

from .models import Session, SessionMessage, SessionStatus, MessageType
from .storage import SessionStorage

logger = logging.getLogger(__name__)


class SessionManager:
    """High-level session management interface."""
    
    def __init__(self, storage: Optional[SessionStorage] = None, data_dir: str = "./data"):
        """Initialize session manager.
        
        Args:
            storage: Session storage instance (creates default if None)
            data_dir: Directory for session data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        if storage is None:
            db_path = self.data_dir / "sessions.db"
            storage = SessionStorage(str(db_path))
        
        self.storage = storage
        self.current_session: Optional[Session] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the session manager."""
        if self._initialized:
            return
        
        logger.info("Initializing session manager...")
        await self.storage.initialize()
        self._initialized = True
        logger.info("Session manager initialized")
    
    async def start_new_session(self, title: Optional[str] = None) -> Session:
        """Start a new conversation session.
        
        Args:
            title: Optional session title
            
        Returns:
            New session instance
        """
        await self.initialize()
        
        # Complete current session if active
        if self.current_session and self.current_session.status == SessionStatus.ACTIVE:
            await self.complete_current_session()
        
        # Create new session
        session_title = title or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        session = Session(
            id=uuid4(),
            title=session_title,
            status=SessionStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Save to storage
        await self.storage.create_session(session)
        self.current_session = session
        
        logger.info(f"Started new session: {session.id} - {session.title}")
        return session
    
    async def get_current_session(self) -> Optional[Session]:
        """Get the current active session.
        
        Returns:
            Current session or None if no active session
        """
        return self.current_session
    
    async def add_message_to_current_session(
        self, 
        content: str, 
        message_type: MessageType,
        metadata: Optional[dict] = None
    ) -> Optional[SessionMessage]:
        """Add a message to the current session.
        
        Args:
            content: Message content
            message_type: Type of message
            metadata: Optional message metadata
            
        Returns:
            Created message or None if no current session
        """
        if not self.current_session:
            logger.warning("No current session to add message to")
            return None
        
        message = SessionMessage(
            id=uuid4(),
            session_id=self.current_session.id,
            content=content,
            message_type=message_type,
            timestamp=datetime.now(),
            metadata=metadata
        )
        
        # Add to session
        self.current_session.add_message(message)
        
        # Save to storage
        await self.storage.add_message(message)
        await self.storage.update_session(self.current_session)
        
        logger.debug(f"Added {message_type.value} message to session {self.current_session.id}")
        return message
    
    async def add_agent_message_to_current_session(self, agent_message) -> Optional[SessionMessage]:
        """Convert and add an AgentMessage to the current session.
        
        Args:
            agent_message: AgentMessage from the voice agent
            
        Returns:
            Created session message or None if no current session
        """
        # Convert AgentMessage to SessionMessage
        message_type_map = {
            "user": MessageType.USER,
            "assistant": MessageType.ASSISTANT,
            "system": MessageType.SYSTEM,
            "tool": MessageType.TOOL
        }
        
        message_type = message_type_map.get(agent_message.message_type, MessageType.SYSTEM)
        
        return await self.add_message_to_current_session(
            content=agent_message.content,
            message_type=message_type,
            metadata=agent_message.metadata
        )
    
    async def complete_current_session(self) -> Optional[Session]:
        """Complete the current session.
        
        Returns:
            Completed session or None if no current session
        """
        if not self.current_session:
            return None
        
        self.current_session.complete()
        await self.storage.update_session(self.current_session)
        
        logger.info(f"Completed session: {self.current_session.id}")
        completed_session = self.current_session
        self.current_session = None
        
        return completed_session
    
    async def get_session(self, session_id: UUID) -> Optional[Session]:
        """Get a session by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session if found, None otherwise
        """
        await self.initialize()
        return await self.storage.get_session(session_id)
    
    async def list_sessions(
        self, 
        limit: int = 50, 
        offset: int = 0,
        status: Optional[SessionStatus] = None
    ) -> List[Session]:
        """List sessions with pagination.
        
        Args:
            limit: Maximum number of sessions
            offset: Number of sessions to skip
            status: Filter by status
            
        Returns:
            List of sessions
        """
        await self.initialize()
        return await self.storage.list_sessions(limit, offset, status)
    
    async def search_sessions(self, query: str, limit: int = 20) -> List[Session]:
        """Search sessions by content.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching sessions
        """
        await self.initialize()
        return await self.storage.search_sessions(query, limit)
    
    async def delete_session(self, session_id: UUID) -> bool:
        """Delete a session.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        await self.initialize()
        
        # Don't delete current session
        if self.current_session and self.current_session.id == session_id:
            logger.warning("Cannot delete current active session")
            return False
        
        return await self.storage.delete_session(session_id)
    
    async def export_session(
        self, 
        session_id: UUID, 
        format: str = "json",
        output_path: Optional[str] = None
    ) -> str:
        """Export a session to file.
        
        Args:
            session_id: Session ID to export
            format: Export format ("json", "txt")
            output_path: Output file path (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        await self.initialize()
        
        session = await self.storage.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = session.created_at.strftime("%Y%m%d_%H%M%S")
            filename = f"session_{timestamp}.{format}"
            output_path = str(self.data_dir / "exports" / filename)
        
        # Create export directory
        export_path = Path(output_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export based on format
        if format.lower() == "json":
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
        
        elif format.lower() == "txt":
            with open(export_path, "w", encoding="utf-8") as f:
                f.write(f"Session: {session.title}\n")
                f.write(f"Created: {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Status: {session.status.value}\n")
                if session.completed_at:
                    f.write(f"Completed: {session.completed_at.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Messages: {session.metadata.total_messages}\n")
                f.write("-" * 50 + "\n\n")
                f.write(session.get_conversation_text())
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported session {session_id} to {output_path}")
        return str(export_path)
    
    async def get_session_statistics(self) -> dict:
        """Get overall session statistics.
        
        Returns:
            Dictionary with session statistics
        """
        await self.initialize()
        
        total_sessions = await self.storage.get_session_count()
        active_sessions = await self.storage.get_session_count(SessionStatus.ACTIVE)
        completed_sessions = await self.storage.get_session_count(SessionStatus.COMPLETED)
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "completed_sessions": completed_sessions,
            "current_session_id": str(self.current_session.id) if self.current_session else None,
            "current_session_title": self.current_session.title if self.current_session else None
        }
    
    async def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Clean up old sessions.
        
        Args:
            days_old: Delete sessions older than this many days
            
        Returns:
            Number of sessions deleted
        """
        await self.initialize()
        return await self.storage.cleanup_old_sessions(days_old, keep_completed=True)
    
    async def replay_session(self, session_id: UUID) -> List[SessionMessage]:
        """Get session messages for replay.
        
        Args:
            session_id: Session ID to replay
            
        Returns:
            List of session messages in chronological order
        """
        await self.initialize()
        
        session = await self.storage.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Return messages sorted by timestamp
        return sorted(session.messages, key=lambda m: m.timestamp)
    
    async def close(self) -> None:
        """Close the session manager."""
        # Complete current session if active
        if self.current_session and self.current_session.status == SessionStatus.ACTIVE:
            await self.complete_current_session()
        
        await self.storage.close()
        logger.info("Session manager closed")