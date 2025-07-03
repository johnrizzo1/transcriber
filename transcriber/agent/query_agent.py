"""Specialized agent for single query processing with memory."""

import asyncio
import logging
import time

from ..config import Settings
from ..memory import MemoryManager
from ..memory.models import QueryResult
from .core import VoiceAgent

logger = logging.getLogger(__name__)


class QueryAgent:
    """Agent specialized for single query processing with memory context."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.voice_agent = VoiceAgent(settings)
        
        # Initialize memory manager if enabled
        if settings.memory.enabled:
            self.memory_manager = MemoryManager(settings.memory)
        else:
            self.memory_manager = None
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the query agent."""
        if self._initialized:
            return
        
        logger.info("Initializing query agent...")
        
        # Initialize voice agent
        await self.voice_agent.initialize()
        
        # Initialize memory manager
        if self.memory_manager:
            await self.memory_manager.initialize()
        
        self._initialized = True
        logger.info("Query agent initialized")
    
    async def process_query(
        self,
        query: str,
        use_memory: bool = True,
        store_interaction: bool = True,
        verbose: bool = False
    ) -> QueryResult:
        """Process a single query with optional memory context."""
        start_time = time.time()
        
        try:
            await self.initialize()
            
            # Retrieve memory context if enabled
            memory_context = None
            if use_memory and self.memory_manager:
                memory_context = await self.memory_manager.retrieve_context(
                    query
                )
                
                if verbose and memory_context.has_relevant_context():
                    logger.info(
                        f"Retrieved {len(memory_context.relevant_memories)} "
                        f"relevant memories"
                    )
            
            # Build enhanced prompt with memory context
            if memory_context and memory_context.has_relevant_context():
                enhanced_prompt = self._build_memory_enhanced_prompt(
                    self.voice_agent.system_prompt,
                    memory_context
                )
            else:
                enhanced_prompt = self.voice_agent.system_prompt
            
            # Process query with enhanced context
            original_prompt = self.voice_agent.system_prompt
            self.voice_agent.system_prompt = enhanced_prompt
            
            try:
                response = await self.voice_agent.process_text_input(query)
            finally:
                # Restore original prompt
                self.voice_agent.system_prompt = original_prompt
            
            # Store interaction in memory if enabled
            if store_interaction and self.memory_manager:
                await self.memory_manager.store_interaction(
                    query=query,
                    response=response,
                    metadata={
                        "processing_time": time.time() - start_time,
                        "memory_context_used": memory_context is not None,
                        "context_memories_count": (
                            len(memory_context.relevant_memories)
                            if memory_context else 0
                        )
                    }
                )
                
                # Wait for background tasks to complete if background enabled
                if self.memory_manager.config.background_processing:
                    # Give background tasks a moment to complete
                    await asyncio.sleep(0.1)
            
            processing_time = time.time() - start_time
            
            return QueryResult(
                query=query,
                response=response,
                memory_context=memory_context,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return QueryResult(
                query=query,
                response=f"Sorry, I encountered an error: {str(e)}",
                memory_context=None,
                processing_time=time.time() - start_time,
                error=str(e)
            )
    
    def _build_memory_enhanced_prompt(
        self, base_prompt: str, memory_context
    ) -> str:
        """Build system prompt enhanced with memory context."""
        if not memory_context or not memory_context.has_relevant_context():
            return base_prompt
        
        context_section = f"""

RELEVANT CONVERSATION HISTORY:
{memory_context.get_context_text()}

Please use this context to provide more personalized and informed responses. 
If the user asks about something mentioned in the conversation history, 
reference it appropriately.
"""
        
        return base_prompt + context_section
    
    async def get_memory_statistics(self) -> dict:
        """Get memory system statistics."""
        if not self.memory_manager:
            return {"memory_enabled": False}
        
        return await self.memory_manager.get_memory_statistics()
    
    async def cleanup(self) -> None:
        """Clean up query agent resources."""
        if self.memory_manager:
            await self.memory_manager.close()
        
        await self.voice_agent.cleanup()
        logger.info("Query agent cleaned up")