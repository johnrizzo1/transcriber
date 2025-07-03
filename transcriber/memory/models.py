"""Data models for memory system."""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import uuid4


@dataclass
class MemoryEntry:
    """Individual memory entry with metadata."""
    id: str
    content: str
    entry_type: str  # "user_query", "assistant_response", "conversation"
    timestamp: datetime
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    
    @classmethod
    def create_user_query(cls, content: str, **metadata) -> "MemoryEntry":
        """Create a user query memory entry."""
        return cls(
            id=str(uuid4()),
            content=content,
            entry_type="user_query",
            timestamp=datetime.now(),
            metadata=metadata
        )
    
    @classmethod
    def create_assistant_response(
        cls, content: str, **metadata
    ) -> "MemoryEntry":
        """Create an assistant response memory entry."""
        return cls(
            id=str(uuid4()),
            content=content,
            entry_type="assistant_response", 
            timestamp=datetime.now(),
            metadata=metadata
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class MemoryContext:
    """Retrieved memory context for a query."""
    relevant_memories: List[MemoryEntry]
    similarity_scores: List[float]
    context_summary: str
    total_memories: int
    query_embedding: Optional[List[float]] = None
    
    def get_context_text(self) -> str:
        """Get formatted context text for prompt inclusion."""
        if not self.relevant_memories:
            return ""
        
        context_lines = ["Previous relevant conversations:"]
        for memory in self.relevant_memories:
            timestamp = memory.timestamp.strftime("%Y-%m-%d %H:%M")
            context_lines.append(
                f"[{timestamp}] {memory.entry_type}: {memory.content}"
            )
        
        return "\n".join(context_lines)
    
    def has_relevant_context(self) -> bool:
        """Check if there's relevant context to include."""
        return len(self.relevant_memories) > 0


@dataclass
class QueryResult:
    """Result of a query processing operation."""
    query: str
    response: str
    memory_context: Optional[MemoryContext]
    processing_time: float
    tokens_used: Optional[int] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "response": self.response,
            "has_memory_context": self.memory_context is not None,
            "context_memories_count": (
                len(self.memory_context.relevant_memories) 
                if self.memory_context else 0
            ),
            "processing_time": self.processing_time,
            "tokens_used": self.tokens_used,
            "error": self.error
        }