"""ChromaDB storage layer for vector memory."""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path

from .models import MemoryEntry

logger = logging.getLogger(__name__)


class ChromaDBStorage:
    """ChromaDB storage layer with optimizations for voice agent."""
    
    def __init__(self, config, embedding_service):
        self.config = config
        self.embedding_service = embedding_service
        self.client = None
        self.collection = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize ChromaDB with persistent storage."""
        if self._initialized:
            return
        
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            # Ensure storage directory exists
            storage_path = Path(self.config.chromadb_path)
            storage_path.mkdir(parents=True, exist_ok=True)
            
            # Configure ChromaDB for local persistence
            self.client = chromadb.PersistentClient(
                path=str(storage_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,  # Privacy-first
                    allow_reset=False
                )
            )
            
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            logger.info(f"ChromaDB initialized at {storage_path}")
            self._initialized = True
            
        except ImportError:
            raise RuntimeError(
                "ChromaDB not installed. Run: pip install chromadb"
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    async def store_memory(self, memory: MemoryEntry) -> None:
        """Store a memory entry in ChromaDB."""
        await self.initialize()
        
        try:
            # Generate embedding if not present
            if not memory.embedding:
                memory.embedding = await self.embedding_service.embed_text(
                    memory.content
                )
            
            # Prepare metadata for ChromaDB
            metadata = {
                "entry_type": memory.entry_type,
                "timestamp": memory.timestamp.isoformat(),
                **memory.metadata
            }
            
            # Add to collection
            self.collection.add(
                embeddings=[memory.embedding],
                documents=[memory.content],
                metadatas=[metadata],
                ids=[memory.id]
            )
            
            logger.debug(f"Stored memory: {memory.id} ({memory.entry_type})")
            
        except Exception as e:
            logger.error(f"Failed to store memory {memory.id}: {e}")
            raise
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar memories using vector similarity."""
        await self.initialize()
        
        try:
            # Build where clause for filtering
            # ChromaDB has limitations with complex nested operators
            # For now, prioritize the passed where clause over time filtering
            where_clause = None
            if where:
                where_clause = where
            elif self.config.context_window_days > 0:
                # Only apply time filtering if no other where clause provided
                cutoff_date = datetime.now() - timedelta(
                    days=self.config.context_window_days
                )
                where_clause = {"timestamp": {"$gte": cutoff_date.isoformat()}}
            
            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "content": doc,
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        # Convert distance to similarity score
                        "score": 1.0 - results["distances"][0][i]
                    })
            
            # Filter by similarity threshold
            filtered_results = [
                r for r in formatted_results 
                if r["score"] >= self.config.similarity_threshold
            ]
            
            logger.debug(f"Found {len(filtered_results)} relevant memories")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    async def get_memory_count(self) -> int:
        """Get total number of stored memories."""
        await self.initialize()
        
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to get memory count: {e}")
            return 0
    
    async def cleanup_old_memories(self, days_old: int) -> int:
        """Clean up memories older than specified days."""
        await self.initialize()
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            # Query old memories
            embedding_dim = self.embedding_service.get_embedding_dimension()
            old_memories = self.collection.query(
                query_embeddings=[[0.0] * embedding_dim],
                n_results=10000,  # Large number to get all
                where={"timestamp": {"$lt": cutoff_date.isoformat()}},
                include=["metadatas"]
            )
            
            if old_memories["ids"] and old_memories["ids"][0]:
                # Delete old memories
                self.collection.delete(ids=old_memories["ids"][0])
                deleted_count = len(old_memories["ids"][0])
                logger.info(f"Cleaned up {deleted_count} old memories")
                return deleted_count
            
            return 0
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
            return 0
    
    async def close(self) -> None:
        """Close ChromaDB connection."""
        # ChromaDB handles cleanup automatically
        logger.info("ChromaDB storage closed")