"""
Ollama LLM client wrapper for the AI agent.
"""

import logging
from collections.abc import AsyncGenerator
from typing import Optional

try:
    import ollama
except ImportError:
    ollama = None
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import Settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """Async wrapper for Ollama client with streaming support."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        if ollama is None:
            raise ImportError("ollama package not available. Install with: pip install ollama")
        self.client = ollama.AsyncClient(host=self.settings.agent.base_url)
        self._model_loaded = False
        
    async def initialize(self) -> None:
        """Initialize the Ollama client and ensure model is available."""
        logger.info("Initializing Ollama client...")
        
        try:
            # Check if model is available
            models = await self.client.list()
            model_names = [m["name"] for m in models["models"]]
            
            if self.settings.agent.model not in model_names:
                logger.warning(f"Model {self.settings.agent.model} not found. Available models: {model_names}")
                # Try to pull the model
                logger.info(f"Pulling model {self.settings.agent.model}...")
                async for progress in await self.client.pull(model=self.settings.agent.model, stream=True):
                    if "status" in progress:
                        logger.debug(f"Pull progress: {progress['status']}")
            
            # Test the model with a simple prompt
            response = await self.client.generate(
                model=self.settings.agent.model,
                prompt="Hello",
                options={"num_predict": 1}
            )
            
            self._model_loaded = True
            logger.info(f"Ollama client initialized with model: {self.settings.agent.model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_response(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        context: Optional[list[dict[str, str]]] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate a response to a prompt.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            context: Previous conversation context
            temperature: Optional temperature override
            
        Returns:
            Generated response text
        """
        if not self._model_loaded:
            raise RuntimeError("Ollama client not initialized")
        
        # Build messages format
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system", 
                "content": system_prompt
            })
        
        # Add context messages
        if context:
            messages.extend(context)
        
        # Add current prompt
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        options = {
            "temperature": temperature or self.settings.agent.temperature,
            "num_ctx": self.settings.agent.max_context_length,
        }
        
        try:
            response = await self.client.chat(
                model=self.settings.agent.model,
                messages=messages,
                options=options
            )
            
            return response["message"]["content"].strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    async def generate_stream(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        context: Optional[list[dict[str, str]]] = None,
        temperature: Optional[float] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response to a prompt.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            context: Previous conversation context
            temperature: Optional temperature override
            
        Yields:
            Response chunks
        """
        if not self._model_loaded:
            raise RuntimeError("Ollama client not initialized")
        
        # Build messages format
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system", 
                "content": system_prompt
            })
        
        # Add context messages
        if context:
            messages.extend(context)
        
        # Add current prompt
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        options = {
            "temperature": temperature or self.settings.agent.temperature,
            "num_ctx": self.settings.agent.max_context_length,
        }
        
        try:
            async for chunk in await self.client.chat(
                model=self.settings.agent.model,
                messages=messages,
                stream=True,
                options=options
            ):
                if chunk.get("message", {}).get("content"):
                    yield chunk["message"]["content"]
                    
        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            raise
    
    async def check_health(self) -> bool:
        """Check if Ollama server is healthy."""
        try:
            models = await self.client.list()
            return True
        except Exception:
            return False


class ConversationManager:
    """Manages conversation context and history."""
    
    def __init__(self, max_context_messages: int = 10):
        self.max_context_messages = max_context_messages
        self.messages: list[dict[str, str]] = []
        
    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.messages.append({
            "role": "user",
            "content": content
        })
        self._trim_context()
        
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation."""
        self.messages.append({
            "role": "assistant", 
            "content": content
        })
        self._trim_context()
        
    def get_context(self) -> list[dict[str, str]]:
        """Get the current conversation context."""
        return self.messages.copy()
        
    def clear_context(self) -> None:
        """Clear the conversation context."""
        self.messages.clear()
        
    def _trim_context(self) -> None:
        """Trim context to stay within limits."""
        if len(self.messages) > self.max_context_messages:
            # Keep the most recent messages
            self.messages = self.messages[-self.max_context_messages:]


class LLMService:
    """High-level LLM service combining Ollama client and conversation management."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OllamaClient(settings)
        self.conversation = ConversationManager()
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize the LLM service."""
        if not self._initialized:
            await self.client.initialize()
            self._initialized = True
            
    async def process_user_input(
        self, 
        user_input: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Process user input and return assistant response.
        
        Args:
            user_input: The user's input text
            system_prompt: Optional system prompt
            
        Returns:
            Assistant response
        """
        if not self._initialized:
            raise RuntimeError("LLM service not initialized")
        
        # Add user input to conversation
        self.conversation.add_user_message(user_input)
        
        # Generate response
        response = await self.client.generate_response(
            prompt=user_input,
            system_prompt=system_prompt,
            context=self.conversation.get_context()[:-1]  # Exclude the current message
        )
        
        # Add response to conversation
        self.conversation.add_assistant_message(response)
        
        return response
    
    async def process_user_input_stream(
        self, 
        user_input: str,
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Process user input and return streaming assistant response.
        
        Args:
            user_input: The user's input text
            system_prompt: Optional system prompt
            
        Yields:
            Assistant response chunks
        """
        if not self._initialized:
            raise RuntimeError("LLM service not initialized")
        
        # Add user input to conversation
        self.conversation.add_user_message(user_input)
        
        # Generate streaming response
        full_response = ""
        async for chunk in self.client.generate_stream(
            prompt=user_input,
            system_prompt=system_prompt,
            context=self.conversation.get_context()[:-1]  # Exclude the current message
        ):
            full_response += chunk
            yield chunk
        
        # Add complete response to conversation
        self.conversation.add_assistant_message(full_response)
    
    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self.conversation.clear_context()
        
    async def health_check(self) -> bool:
        """Check if the LLM service is healthy."""
        return await self.client.check_health()