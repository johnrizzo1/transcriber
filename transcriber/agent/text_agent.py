"""
Text-only agent for testing without speech components.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from ..config import Settings
from .llm import LLMService

logger = logging.getLogger(__name__)


class TextAgentState(Enum):
    """Text agent state enumeration."""
    IDLE = "idle"
    PROCESSING = "processing"
    RESPONDING = "responding"
    ERROR = "error"


@dataclass
class TextMessage:
    """Message structure for text agent communication."""
    content: str
    timestamp: datetime
    message_type: str  # "user", "assistant", "system"
    metadata: Optional[dict[str, Any]] = None


class TextOnlyAgent:
    """Text-only agent without speech components."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm_service = LLMService(settings)
        
        self.state = TextAgentState.IDLE
        self.conversation_history: list[TextMessage] = []
        self._state_callbacks: list = []
        
        # System prompt for the agent
        self.system_prompt = """You are a helpful AI assistant. 
You can help with various tasks including answering questions, providing information, 
and general conversation. Keep your responses clear and helpful."""
        
    async def initialize(self) -> None:
        """Initialize the text agent."""
        logger.info("Initializing text-only agent...")
        
        try:
            # Initialize LLM service only
            await self.llm_service.initialize()
            logger.info("Text-only agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize text agent: {e}")
            self.state = TextAgentState.ERROR
            raise
    
    def add_state_callback(self, callback) -> None:
        """Add a callback to be called when agent state changes."""
        self._state_callbacks.append(callback)
        
    def _set_state(self, new_state: TextAgentState) -> None:
        """Set agent state and notify callbacks."""
        if new_state != self.state:
            old_state = self.state
            self.state = new_state
            logger.debug(f"Text agent state changed: {old_state} -> {new_state}")
            
            # Notify callbacks
            for callback in self._state_callbacks:
                try:
                    callback(old_state, new_state)
                except Exception as e:
                    logger.error(f"Error in state callback: {e}")
    
    async def process_text_input(self, text_input: str) -> str:
        """
        Process text input and return response.
        
        Args:
            text_input: User text input
            
        Returns:
            Agent response
        """
        try:
            self._set_state(TextAgentState.PROCESSING)
            
            # Add user message to history
            user_message = TextMessage(
                content=text_input,
                timestamp=datetime.now(),
                message_type="user"
            )
            self.conversation_history.append(user_message)
            
            logger.info(f"User input: {text_input}")
            
            # Generate response
            self._set_state(TextAgentState.RESPONDING)
            
            response = await self.llm_service.process_user_input(
                text_input, 
                self.system_prompt
            )
            
            # Add assistant message to history
            assistant_message = TextMessage(
                content=response,
                timestamp=datetime.now(),
                message_type="assistant"
            )
            self.conversation_history.append(assistant_message)
            
            logger.info(f"Agent responded: {response}")
            
            self._set_state(TextAgentState.IDLE)
            return response
            
        except Exception as e:
            logger.error(f"Error processing text input: {e}")
            self._set_state(TextAgentState.ERROR)
            error_response = f"Sorry, I encountered an error: {e!s}"
            self._set_state(TextAgentState.IDLE)
            return error_response
    
    async def process_text_input_stream(self, text_input: str) -> AsyncGenerator[str, None]:
        """
        Process text input and return streaming response.
        
        Args:
            text_input: User text input
            
        Yields:
            Response chunks
        """
        try:
            self._set_state(TextAgentState.PROCESSING)
            
            # Add user message to history
            user_message = TextMessage(
                content=text_input,
                timestamp=datetime.now(),
                message_type="user"
            )
            self.conversation_history.append(user_message)
            
            logger.info(f"User input: {text_input}")
            
            # Generate streaming response
            self._set_state(TextAgentState.RESPONDING)
            
            full_response = ""
            async for response_chunk in self.llm_service.process_user_input_stream(
                text_input, 
                self.system_prompt
            ):
                full_response += response_chunk
                yield response_chunk
            
            # Add assistant message to history
            assistant_message = TextMessage(
                content=full_response,
                timestamp=datetime.now(),
                message_type="assistant"
            )
            self.conversation_history.append(assistant_message)
            
            logger.info(f"Agent responded: {full_response}")
            
            self._set_state(TextAgentState.IDLE)
            
        except Exception as e:
            logger.error(f"Error processing text input: {e}")
            self._set_state(TextAgentState.ERROR)
            yield f"Sorry, I encountered an error: {e!s}"
            self._set_state(TextAgentState.IDLE)
    
    def get_conversation_history(self) -> list[TextMessage]:
        """Get the conversation history."""
        return self.conversation_history.copy()
    
    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self.conversation_history.clear()
        self.llm_service.clear_conversation()
        logger.info("Conversation history cleared")
    
    async def health_check(self) -> dict[str, bool]:
        """Check the health of the agent."""
        health_status = {
            "llm": False,
            "overall": False
        }
        
        try:
            # Check LLM service
            health_status["llm"] = await self.llm_service.health_check()
            health_status["overall"] = health_status["llm"]
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            
        return health_status
    
    async def cleanup(self) -> None:
        """Clean up agent resources."""
        logger.info("Cleaning up text-only agent...")
        # LLM service doesn't require explicit cleanup
        self._set_state(TextAgentState.IDLE)
        logger.info("Text-only agent cleanup complete")


async def run_interactive_chat(settings: Settings) -> None:
    """Run an interactive chat session."""
    from rich.console import Console
    
    console = Console()
    agent = TextOnlyAgent(settings)
    
    try:
        await agent.initialize()
        
        console.print("[yellow]Text-only chat mode initialized![/yellow]")
        console.print("[dim]Type 'quit' to exit[/dim]")
        
        while True:
            try:
                user_input = input("\n🎤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if not user_input:
                    continue
                
                console.print("🤖 Agent: ", end="", style="bold green")
                
                # Stream response
                async for chunk in agent.process_text_input_stream(user_input):
                    console.print(chunk, end="", style="green")
                
                console.print()  # New line after response
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
        
        console.print("[yellow]Goodbye![/yellow]")
        await agent.cleanup()
        
    except Exception as e:
        console.print(f"[red]Failed to initialize: {e}[/red]")
        import traceback
        traceback.print_exc()