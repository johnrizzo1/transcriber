"""
SQLite storage implementation for session management.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import UUID

import aiosqlite

from .models import Session, SessionMessage, SessionMetadata, SessionStatus, MessageType

logger = logging.getLogger(__name__)


class SessionStorage:
    """SQLite-based session storage."""
    
    def __init__(self, db_path: str = "./data/sessions.db"):
        """Initialize session storage.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the database schema."""
        if self._initialized:
            return
        
        logger.info(f"Initializing session storage at {self.db_path}")
        
        async with aiosqlite.connect(self.db_path) as db:
            # Create sessions table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    metadata TEXT NOT NULL
                )
            """)
            
            # Create messages table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            """)
            
            # Create indexes for better performance
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_created_at 
                ON sessions (created_at)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_status 
                ON sessions (status)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session_id 
                ON messages (session_id)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp 
                ON messages (timestamp)
            """)
            
            await db.commit()
        
        self._initialized = True
        logger.info("Session storage initialized successfully")
    
    async def create_session(self, session: Session) -> None:
        """Create a new session.
        
        Args:
            session: Session to create
        """
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO sessions (
                    id, title, status, created_at, updated_at, 
                    completed_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                str(session.id),
                session.title,
                session.status.value,
                session.created_at.isoformat(),
                session.updated_at.isoformat(),
                session.completed_at.isoformat() if session.completed_at else None,
                json.dumps(session.metadata.to_dict())
            ))
            
            # Insert messages
            for message in session.messages:
                await self._insert_message(db, message)
            
            await db.commit()
        
        logger.info(f"Created session {session.id}")
    
    async def get_session(self, session_id: UUID) -> Optional[Session]:
        """Get a session by ID.
        
        Args:
            session_id: Session ID to retrieve
            
        Returns:
            Session if found, None otherwise
        """
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            # Get session data
            async with db.execute("""
                SELECT id, title, status, created_at, updated_at, 
                       completed_at, metadata
                FROM sessions WHERE id = ?
            """, (str(session_id),)) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None
            
            # Parse session data
            session = Session(
                id=UUID(row[0]),
                title=row[1],
                status=SessionStatus(row[2]),
                created_at=datetime.fromisoformat(row[3]),
                updated_at=datetime.fromisoformat(row[4]),
                completed_at=datetime.fromisoformat(row[5]) if row[5] else None,
                metadata=SessionMetadata.from_dict(json.loads(row[6]))
            )
            
            # Get messages
            async with db.execute("""
                SELECT id, session_id, content, message_type, timestamp, metadata
                FROM messages WHERE session_id = ?
                ORDER BY timestamp ASC
            """, (str(session_id),)) as cursor:
                async for msg_row in cursor:
                    message = SessionMessage(
                        id=UUID(msg_row[0]),
                        session_id=UUID(msg_row[1]),
                        content=msg_row[2],
                        message_type=MessageType(msg_row[3]),
                        timestamp=datetime.fromisoformat(msg_row[4]),
                        metadata=json.loads(msg_row[5]) if msg_row[5] else None
                    )
                    session.messages.append(message)
        
        return session
    
    async def update_session(self, session: Session) -> None:
        """Update an existing session.
        
        Args:
            session: Session to update
        """
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            # Update session
            await db.execute("""
                UPDATE sessions SET
                    title = ?, status = ?, updated_at = ?, 
                    completed_at = ?, metadata = ?
                WHERE id = ?
            """, (
                session.title,
                session.status.value,
                session.updated_at.isoformat(),
                session.completed_at.isoformat() if session.completed_at else None,
                json.dumps(session.metadata.to_dict()),
                str(session.id)
            ))
            
            await db.commit()
        
        logger.debug(f"Updated session {session.id}")
    
    async def add_message(self, message: SessionMessage) -> None:
        """Add a message to a session.
        
        Args:
            message: Message to add
        """
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            await self._insert_message(db, message)
            
            # Update session timestamp
            await db.execute("""
                UPDATE sessions SET updated_at = ?
                WHERE id = ?
            """, (
                datetime.now().isoformat(),
                str(message.session_id)
            ))
            
            await db.commit()
        
        logger.debug(f"Added message to session {message.session_id}")
    
    async def list_sessions(
        self, 
        limit: int = 50, 
        offset: int = 0,
        status: Optional[SessionStatus] = None
    ) -> List[Session]:
        """List sessions with pagination.
        
        Args:
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip
            status: Filter by session status
            
        Returns:
            List of sessions (without messages for performance)
        """
        await self.initialize()
        
        sessions = []
        
        async with aiosqlite.connect(self.db_path) as db:
            query = """
                SELECT id, title, status, created_at, updated_at, 
                       completed_at, metadata
                FROM sessions
            """
            params = []
            
            if status:
                query += " WHERE status = ?"
                params.append(status.value)
            
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            async with db.execute(query, params) as cursor:
                async for row in cursor:
                    session = Session(
                        id=UUID(row[0]),
                        title=row[1],
                        status=SessionStatus(row[2]),
                        created_at=datetime.fromisoformat(row[3]),
                        updated_at=datetime.fromisoformat(row[4]),
                        completed_at=datetime.fromisoformat(row[5]) if row[5] else None,
                        metadata=SessionMetadata.from_dict(json.loads(row[6]))
                    )
                    sessions.append(session)
        
        return sessions
    
    async def delete_session(self, session_id: UUID) -> bool:
        """Delete a session and all its messages.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if session was deleted, False if not found
        """
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            # Delete messages first (foreign key constraint)
            await db.execute("""
                DELETE FROM messages WHERE session_id = ?
            """, (str(session_id),))
            
            # Delete session
            cursor = await db.execute("""
                DELETE FROM sessions WHERE id = ?
            """, (str(session_id),))
            
            await db.commit()
            
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted session {session_id}")
            
            return deleted
    
    async def search_sessions(
        self, 
        query: str, 
        limit: int = 20
    ) -> List[Session]:
        """Search sessions by title or message content.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching sessions
        """
        await self.initialize()
        
        sessions = []
        search_term = f"%{query}%"
        
        async with aiosqlite.connect(self.db_path) as db:
            # Search in session titles and message content
            async with db.execute("""
                SELECT DISTINCT s.id, s.title, s.status, s.created_at, 
                       s.updated_at, s.completed_at, s.metadata
                FROM sessions s
                LEFT JOIN messages m ON s.id = m.session_id
                WHERE s.title LIKE ? OR m.content LIKE ?
                ORDER BY s.created_at DESC
                LIMIT ?
            """, (search_term, search_term, limit)) as cursor:
                async for row in cursor:
                    session = Session(
                        id=UUID(row[0]),
                        title=row[1],
                        status=SessionStatus(row[2]),
                        created_at=datetime.fromisoformat(row[3]),
                        updated_at=datetime.fromisoformat(row[4]),
                        completed_at=datetime.fromisoformat(row[5]) if row[5] else None,
                        metadata=SessionMetadata.from_dict(json.loads(row[6]))
                    )
                    sessions.append(session)
        
        return sessions
    
    async def get_session_count(self, status: Optional[SessionStatus] = None) -> int:
        """Get total number of sessions.
        
        Args:
            status: Filter by session status
            
        Returns:
            Number of sessions
        """
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            if status:
                async with db.execute("""
                    SELECT COUNT(*) FROM sessions WHERE status = ?
                """, (status.value,)) as cursor:
                    row = await cursor.fetchone()
            else:
                async with db.execute("""
                    SELECT COUNT(*) FROM sessions
                """) as cursor:
                    row = await cursor.fetchone()
            
            return row[0] if row else 0
    
    async def cleanup_old_sessions(
        self, 
        days_old: int = 30,
        keep_completed: bool = True
    ) -> int:
        """Clean up old sessions.
        
        Args:
            days_old: Delete sessions older than this many days
            keep_completed: Whether to keep completed sessions
            
        Returns:
            Number of sessions deleted
        """
        await self.initialize()
        
        cutoff_date = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        cutoff_date = cutoff_date.replace(
            day=cutoff_date.day - days_old
        ).isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            query = """
                DELETE FROM sessions 
                WHERE created_at < ?
            """
            params = [cutoff_date]
            
            if keep_completed:
                query += " AND status != ?"
                params.append(SessionStatus.COMPLETED.value)
            
            cursor = await db.execute(query, params)
            await db.commit()
            
            deleted_count = cursor.rowcount
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old sessions")
            
            return deleted_count
    
    async def _insert_message(self, db: aiosqlite.Connection, message: SessionMessage) -> None:
        """Insert a message into the database.
        
        Args:
            db: Database connection
            message: Message to insert
        """
        await db.execute("""
            INSERT INTO messages (
                id, session_id, content, message_type, timestamp, metadata
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            str(message.id),
            str(message.session_id),
            message.content,
            message.message_type.value,
            message.timestamp.isoformat(),
            json.dumps(message.metadata) if message.metadata else None
        ))
    
    async def close(self) -> None:
        """Close the storage (cleanup if needed)."""
        # SQLite connections are closed automatically with aiosqlite
        logger.info("Session storage closed")