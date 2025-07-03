"""Main memory manager coordinating all memory operations."""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
import asyncio

from .models import MemoryEntry, MemoryContext
from .embeddings import EmbeddingService
from .storage import ChromaDBStorage

logger = logging.getLogger(__name__)


class MemoryManager:
    """High-level memory management interface."""
    
    def __init__(self, config):
        self.config = config
        self.embedding_service = EmbeddingService(config)
        self.storage = ChromaDBStorage(config, self.embedding_service)
        self._initialized = False
        self._background_tasks = set()
    
    async def initialize(self) -> None:
        """Initialize memory manager and all components."""
        if self._initialized:
            return
        
        if not self.config.enabled:
            logger.info("Memory system disabled by configuration")
            return
        
        logger.info("Initializing memory manager...")
        
        try:
            # Initialize components
            await self.embedding_service.initialize()
            await self.storage.initialize()
            
            # Start background cleanup if enabled
            if self.config.auto_cleanup_enabled:
                self._start_background_cleanup()
            
            self._initialized = True
            logger.info("Memory manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory manager: {e}")
            raise
    
    async def store_interaction(
        self,
        query: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store a query-response interaction in memory."""
        if not self.config.enabled:
            return
        
        await self.initialize()
        
        try:
            base_metadata = metadata or {}
            base_metadata.update({
                "interaction_id": f"interaction_{datetime.now().timestamp()}",
                "source": "query_command"
            })
            
            # Create memory entries
            user_memory = MemoryEntry.create_user_query(query, **base_metadata)
            assistant_memory = MemoryEntry.create_assistant_response(
                response, **base_metadata
            )
            
            # Store in background if configured
            if self.config.background_processing:
                self._schedule_background_storage(
                    [user_memory, assistant_memory]
                )
            else:
                await self.storage.store_memory(user_memory)
                await self.storage.store_memory(assistant_memory)
            
            logger.debug("Stored interaction in memory")
            
        except Exception as e:
            logger.error(f"Failed to store interaction: {e}")
    
    async def retrieve_context(
        self,
        query: str,
        max_memories: Optional[int] = None
    ) -> MemoryContext:
        """Retrieve relevant memory context for a query."""
        if not self.config.enabled:
            return MemoryContext([], [], "", 0)
        
        await self.initialize()
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.embed_text(query)
            
            # Search for similar memories
            max_results = max_memories or self.config.max_context_memories
            search_results = await self.storage.similarity_search(
                query_embedding=query_embedding,
                n_results=max_results * 2,  # Get more to allow for ranking
                where={
                    "entry_type": {"$in": ["user_query", "assistant_response"]}
                }
            )
            
            # Convert to memory entries and rank
            memories = []
            scores = []
            
            for result in search_results:
                memory_data = {
                    "id": result["id"],
                    "content": result["content"],
                    "entry_type": result["metadata"]["entry_type"],
                    "timestamp": datetime.fromisoformat(
                        result["metadata"]["timestamp"]
                    ),
                    "metadata": {
                        k: v for k, v in result["metadata"].items() 
                        if k not in ["entry_type", "timestamp"]
                    }
                }
                
                memory = MemoryEntry(**memory_data)
                memories.append(memory)
                scores.append(result["score"])
            
            # Rank and select top memories
            ranked_memories = self._rank_memories(
                query, memories, scores
            )[:max_results]
            ranked_scores = scores[:len(ranked_memories)]
            
            # Generate context summary
            context_summary = self._generate_context_summary(ranked_memories)
            
            return MemoryContext(
                relevant_memories=ranked_memories,
                similarity_scores=ranked_scores,
                context_summary=context_summary,
                total_memories=len(ranked_memories),
                query_embedding=query_embedding
            )
            
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return MemoryContext([], [], "", 0)
    
    def _rank_memories(
        self,
        query: str,
        memories: List[MemoryEntry],
        scores: List[float]
    ) -> List[MemoryEntry]:
        """Rank memories by relevance, recency, and importance."""
        if not memories:
            return []
        
        ranked_items = []
        
        for memory, score in zip(memories, scores):
            # Calculate recency boost (more recent = higher boost)
            days_old = (datetime.now() - memory.timestamp).days
            recency_boost = max(0, 1.0 - (days_old / 30.0))  # Decays over 30 days
            
            # Calculate importance boost (user queries about personal info)
            importance_boost = 0.0
            personal_keywords = ["my name", "i am", "i'm", "my", "me"]
            if any(keyword in memory.content.lower() 
                   for keyword in personal_keywords):
                importance_boost = 0.3
            
            # Calculate conversation pair boost (query-response pairs)
            conversation_boost = 0.0
            if memory.entry_type == "assistant_response":
                conversation_boost = 0.2
            
            # Final score
            final_score = score * (
                1.0 + recency_boost + importance_boost + conversation_boost
            )
            ranked_items.append((memory, final_score))
        
        # Sort by final score
        ranked_items.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, _ in ranked_items]
    
    def _generate_context_summary(self, memories: List[MemoryEntry]) -> str:
        """Generate a summary of the memory context."""
        if not memories:
            return "No relevant context found."
        
        memory_types = {}
        for memory in memories:
            memory_types[memory.entry_type] = (
                memory_types.get(memory.entry_type, 0) + 1
            )
        
        summary_parts = []
        if memory_types.get("user_query", 0) > 0:
            summary_parts.append(f"{memory_types['user_query']} previous queries")
        if memory_types.get("assistant_response", 0) > 0:
            summary_parts.append(
                f"{memory_types['assistant_response']} previous responses"
            )
        
        return (
            f"Found {len(memories)} relevant memories: "
            f"{', '.join(summary_parts)}"
        )
    
    def _schedule_background_storage(self, memories: List[MemoryEntry]) -> None:
        """Schedule memory storage in background."""
        task = asyncio.create_task(self._background_store_memories(memories))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
    
    async def _background_store_memories(self, memories: List[MemoryEntry]) -> None:
        """Store memories in background."""
        try:
            for memory in memories:
                await self.storage.store_memory(memory)
        except Exception as e:
            logger.error(f"Background storage failed: {e}")
    
    def _start_background_cleanup(self) -> None:
        """Start background cleanup task."""
        task = asyncio.create_task(self._background_cleanup_loop())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
    
    async def _background_cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval_hours * 3600)
                deleted_count = await self.storage.cleanup_old_memories(
                    self.config.memory_retention_days
                )
                if deleted_count > 0:
                    logger.info(
                        f"Background cleanup removed {deleted_count} old memories"
                    )
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        if not self.config.enabled:
            return {"enabled": False}
        
        await self.initialize()
        
        try:
            total_memories = await self.storage.get_memory_count()
            
            return {
                "enabled": True,
                "total_memories": total_memories,
                "embedding_model": self.config.embedding_model,
                "embedding_strategy": self.embedding_service.strategy,
                "cache_size": len(self.embedding_service._cache),
                "storage_path": self.config.chromadb_path
            }
        except Exception as e:
            logger.error(f"Failed to get memory statistics: {e}")
            return {"enabled": True, "error": str(e)}
    
    async def cleanup_old_memories(self, days_old: Optional[int] = None) -> int:
        """Manually trigger cleanup of old memories."""
        if not self.config.enabled:
            return 0
        
        await self.initialize()
        days = days_old or self.config.memory_retention_days
        return await self.storage.cleanup_old_memories(days)
    
    async def close(self) -> None:
        """Close memory manager and cleanup resources."""
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Close storage
        await self.storage.close()
        
        # Clear caches
        self.embedding_service.clear_cache()
        
        logger.info("Memory manager closed")