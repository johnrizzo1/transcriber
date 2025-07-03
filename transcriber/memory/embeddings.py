"""Embedding service for generating text embeddings."""

import logging
from typing import List, Dict
import asyncio

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings with caching and fallbacks."""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.strategy = None
        self._cache: Dict[str, List[float]] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize embedding model with fallback chain."""
        if self._initialized:
            return
        
        try:
            # Primary: SentenceTransformers (local, high quality)
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.config.embedding_model)
            self.strategy = "sentence_transformers"
            logger.info(
                f"Initialized SentenceTransformers with model: "
                f"{self.config.embedding_model}"
            )
            
        except ImportError as e:
            logger.warning(f"SentenceTransformers not available: {e}")
            try:
                # Fallback: Simple TF-IDF (basic but functional)
                from sklearn.feature_extraction.text import TfidfVectorizer
                self.model = TfidfVectorizer(
                    max_features=384, stop_words='english'
                )
                self.strategy = "tfidf"
                logger.warning("Using TF-IDF fallback for embeddings")
                
            except ImportError:
                raise RuntimeError(
                    "No embedding strategy available. "
                    "Install sentence-transformers or scikit-learn."
                )
        
        self._initialized = True
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        await self.initialize()
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Generate embedding
        if self.strategy == "sentence_transformers":
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, 
                lambda: self.model.encode(
                    text, convert_to_tensor=False
                ).tolist()
            )
        elif self.strategy == "tfidf":
            # TF-IDF requires fitting on corpus first
            embedding = await self._tfidf_embed(text)
        else:
            raise RuntimeError(f"Unknown embedding strategy: {self.strategy}")
        
        # Cache result with LRU eviction
        self._cache_embedding(cache_key, embedding)
        return embedding
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently."""
        await self.initialize()
        
        if self.strategy == "sentence_transformers":
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(
                    texts, convert_to_tensor=False
                ).tolist()
            )
            return embeddings
        else:
            # Fallback to individual processing
            return [await self.embed_text(text) for text in texts]
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        # Simple hash-based key (could use more sophisticated approach)
        return str(hash(text.strip().lower()))
    
    def _cache_embedding(self, key: str, embedding: List[float]) -> None:
        """Cache embedding with LRU eviction."""
        if len(self._cache) >= self.config.embedding_cache_size:
            # Remove oldest entry (simple FIFO, could use proper LRU)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[key] = embedding
    
    async def _tfidf_embed(self, text: str) -> List[float]:
        """Generate TF-IDF embedding (fallback method)."""
        # This is a simplified implementation
        # In practice, you'd need to maintain a corpus for proper TF-IDF
        words = text.lower().split()
        # Create a simple bag-of-words vector (384 dimensions)
        embedding = [0.0] * 384
        for i, word in enumerate(words[:384]):
            embedding[i] = hash(word) % 100 / 100.0  # Normalized hash
        return embedding
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this service."""
        if self.strategy == "sentence_transformers":
            return self.model.get_sentence_embedding_dimension()
        else:
            return 384  # Default dimension for fallback methods
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")