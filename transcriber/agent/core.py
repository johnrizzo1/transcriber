"""
Core agent orchestrator that coordinates between speech processing and LLM.
"""

import logging
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from ..audio.stt import STTProcessor
from ..config import Settings
from ..session.manager import SessionManager
from ..session.models import MessageType
from ..tools import get_registry, initialize_tools
from .llm import LLMService

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent state enumeration."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    ERROR = "error"


@dataclass
class AgentMessage:
    """Message structure for agent communication."""
    content: str
    timestamp: datetime
    message_type: str  # "user", "assistant", "system", "tool"
    metadata: Optional[dict[str, Any]] = None


class VoiceAgent:
    """Main voice agent that orchestrates all components."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm_service = LLMService(settings)
        self.stt_processor = STTProcessor(settings)
        
        self.state = AgentState.IDLE
        self.conversation_history: list[AgentMessage] = []
        self._running = False
        self._state_callbacks: list = []
        
        # Tool system
        self.tool_registry = get_registry()
        self._tools_initialized = False
        
        # Session management
        self.session_manager: Optional[SessionManager] = None
        self._session_enabled = settings.session.enabled
        
        # System prompt for the agent
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt including available tools."""
        base_prompt = """Your name is ragnasty and you are a helpful AI assistant with a voice interface.
You can help with various tasks including answering questions, providing information,
and executing commands. You are particularly good at transcribing your conversations when asked.
Keep your responses conversational and concise since they will be spoken aloud. If you need clarification, ask directly.

You have access to various tools that you can use to help users. When you need to use a tool,
format your response as: TOOL_CALL: tool_name(param1="value1", param2="value2")

Available tools:"""
        
        if self._tools_initialized:
            tools_info = []
            for tool_name, tool in self.tool_registry.get_all().items():
                params = ", ".join([
                    f"{p.name}: {p.description}"
                    for p in tool.parameters
                ])
                tools_info.append(f"- {tool_name}: {tool.description}")
                if params:
                    tools_info.append(f"  Parameters: {params}")
            
            if tools_info:
                base_prompt += "\n" + "\n".join(tools_info)
        else:
            base_prompt += "\n(Tools will be available after initialization)"
        
        return base_prompt
    
    async def _initialize_tools(self) -> None:
        """Initialize the tool system."""
        if not self._tools_initialized:
            logger.info("Initializing tool system...")
            try:
                discovered = initialize_tools()
                logger.info(f"Discovered {len(discovered)} tools: {discovered}")
                self._tools_initialized = True
                # Update system prompt with available tools
                self.system_prompt = self._build_system_prompt()
            except Exception as e:
                logger.error(f"Failed to initialize tools: {e}")
                raise
        
    async def initialize(self) -> None:
        """Initialize all agent components."""
        logger.info("Initializing voice agent...")
        
        try:
            # Initialize tool system first
            await self._initialize_tools()
            
            # Initialize LLM service
            await self.llm_service.initialize()
            
            # Initialize STT processor
            await self.stt_processor.initialize()
            
            # Initialize session management if enabled
            if self._session_enabled:
                await self._initialize_session_manager()
            
            logger.info("Voice agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize voice agent: {e}")
            self.state = AgentState.ERROR
            raise
    
    async def _initialize_session_manager(self) -> None:
        """Initialize the session management system."""
        logger.info("Initializing session management...")
        
        try:
            # Create session manager with configured data directory
            data_dir = self.settings.data_dir
            self.session_manager = SessionManager(data_dir=data_dir)
            await self.session_manager.initialize()
            
            # Auto-start session if configured
            if self.settings.session.auto_start_session:
                await self.session_manager.start_new_session()
                logger.info("Auto-started new session")
            
        except Exception as e:
            logger.error(f"Failed to initialize session manager: {e}")
            # Don't fail agent initialization for session issues
            self.session_manager = None
    
    def add_state_callback(self, callback) -> None:
        """Add a callback to be called when agent state changes."""
        self._state_callbacks.append(callback)
        
    def _set_state(self, new_state: AgentState) -> None:
        """Set agent state and notify callbacks."""
        if new_state != self.state:
            old_state = self.state
            self.state = new_state
            logger.debug(f"Agent state changed: {old_state} -> {new_state}")
            
            # Notify callbacks
            for callback in self._state_callbacks:
                try:
                    callback(old_state, new_state)
                except Exception as e:
                    logger.error(f"Error in state callback: {e}")
    
    async def process_audio_segments(
        self, 
        audio_segments: AsyncGenerator
    ) -> AsyncGenerator[str, None]:
        """
        Process audio segments and yield text responses.
        
        Args:
            audio_segments: Async generator of audio segments from VAD
            
        Yields:
            Text responses from the agent
        """
        self._set_state(AgentState.LISTENING)
        
        async for segment in audio_segments:
            try:
                self._set_state(AgentState.PROCESSING)
                
                # Transcribe audio segment
                async for transcript in self.stt_processor.process_audio_segments(self._single_segment(segment)):
                    if transcript.strip():
                        # Add user message to history
                        user_message = AgentMessage(
                            content=transcript,
                            timestamp=datetime.now(),
                            message_type="user"
                        )
                        self.conversation_history.append(user_message)
                        
                        # Save to session if enabled
                        if self.session_manager:
                            await self.session_manager.add_agent_message_to_current_session(user_message)
                        
                        logger.info(f"User said: {transcript}")
                        
                        # Generate response
                        self._set_state(AgentState.RESPONDING)
                        
                        full_response = ""
                        async for response_chunk in self.llm_service.process_user_input_stream(
                            transcript,
                            self.system_prompt
                        ):
                            full_response += response_chunk
                            yield response_chunk
                        
                        # Parse and execute any tool calls
                        final_response = await self._parse_and_execute_tools(full_response)
                        
                        # If tools were executed, yield the additional content
                        if final_response != full_response:
                            additional_content = final_response[len(full_response):]
                            if additional_content:
                                yield additional_content
                        
                        # Add assistant message to history
                        assistant_message = AgentMessage(
                            content=final_response,
                            timestamp=datetime.now(),
                            message_type="assistant"
                        )
                        self.conversation_history.append(assistant_message)
                        
                        logger.info(f"Agent responded: {final_response}")
                        
                self._set_state(AgentState.LISTENING)
                
            except Exception as e:
                logger.error(f"Error processing audio segment: {e}")
                self._set_state(AgentState.ERROR)
                yield f"Sorry, I encountered an error: {e!s}"
                self._set_state(AgentState.LISTENING)
    
    async def process_text_input(self, text_input: str) -> str:
        """
        Process text input directly (for testing or fallback).
        
        Args:
            text_input: User text input
            
        Returns:
            Agent response
        """
        try:
            self._set_state(AgentState.PROCESSING)
            
            # Add user message to history
            user_message = AgentMessage(
                content=text_input,
                timestamp=datetime.now(),
                message_type="user"
            )
            self.conversation_history.append(user_message)
            
            # Save to session if enabled
            if self.session_manager:
                await self.session_manager.add_agent_message_to_current_session(user_message)
            
            logger.info(f"User input: {text_input}")
            
            # Generate response
            self._set_state(AgentState.RESPONDING)
            
            response = await self.llm_service.process_user_input(
                text_input,
                self.system_prompt
            )
            
            # Parse and execute any tool calls
            final_response = await self._parse_and_execute_tools(response)
            
            # Add assistant message to history
            assistant_message = AgentMessage(
                content=final_response,
                timestamp=datetime.now(),
                message_type="assistant"
            )
            self.conversation_history.append(assistant_message)
            
            # Save to session if enabled
            if self.session_manager:
                await self.session_manager.add_agent_message_to_current_session(assistant_message)
            
            logger.info(f"Agent responded: {final_response}")
            
            self._set_state(AgentState.IDLE)
            return final_response
            
        except Exception as e:
            logger.error(f"Error processing text input: {e}")
            self._set_state(AgentState.ERROR)
            error_response = f"Sorry, I encountered an error: {e!s}"
            self._set_state(AgentState.IDLE)
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
            self._set_state(AgentState.PROCESSING)

            # Add user message to history
            user_message = AgentMessage(
                content=text_input,
                timestamp=datetime.now(),
                message_type="user"
            )
            self.conversation_history.append(user_message)
            
            # Save to session if enabled
            if self.session_manager:
                await self.session_manager.add_agent_message_to_current_session(user_message)
            
            logger.info(f"User input: {text_input}")
            
            # Generate streaming response
            self._set_state(AgentState.RESPONDING)
            
            full_response = ""
            async for response_chunk in self.llm_service.process_user_input_stream(
                text_input,
                self.system_prompt
            ):
                full_response += response_chunk
                yield response_chunk
            
            # Parse and execute any tool calls
            final_response = await self._parse_and_execute_tools(full_response)
            
            # If tools were executed, yield the additional content
            if final_response != full_response:
                additional_content = final_response[len(full_response):]
                if additional_content:
                    yield additional_content
            
            # Add assistant message to history
            assistant_message = AgentMessage(
                content=final_response,
                timestamp=datetime.now(),
                message_type="assistant"
            )
            self.conversation_history.append(assistant_message)
            
            # Save to session if enabled
            if self.session_manager:
                await self.session_manager.add_agent_message_to_current_session(assistant_message)
            
            logger.info(f"Agent responded: {final_response}")
            
            self._set_state(AgentState.IDLE)
            
        except Exception as e:
            logger.error(f"Error processing text input: {e}")
            self._set_state(AgentState.ERROR)
            yield f"Sorry, I encountered an error: {e!s}"
            self._set_state(AgentState.IDLE)
    
    def get_conversation_history(self) -> list[AgentMessage]:
        """Get the conversation history."""
        return self.conversation_history.copy()
    
    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self.conversation_history.clear()
        self.llm_service.clear_conversation()
        logger.info("Conversation history cleared")
    
    async def _parse_and_execute_tools(self, response: str) -> str:
        """Parse tool calls from response and execute them."""
        if not self._tools_initialized:
            return response
        
        # Look for tool calls in the format: TOOL_CALL: tool_name(param1="value1", param2="value2")
        tool_pattern = r'TOOL_CALL:\s*(\w+)\((.*?)\)'
        matches = re.findall(tool_pattern, response)
        
        if not matches:
            return response
        
        # Execute each tool call
        modified_response = response
        for tool_name, params_str in matches:
            try:
                # Parse parameters
                params = self._parse_tool_params(params_str)
                
                # Execute tool
                result = await self.tool_registry.execute_tool(tool_name, **params)
                
                # Create tool message
                tool_message = AgentMessage(
                    content=f"Tool {tool_name} executed: {result.output if result.success else result.error}",
                    timestamp=datetime.now(),
                    message_type="tool",
                    metadata={"tool_name": tool_name, "success": result.success}
                )
                self.conversation_history.append(tool_message)
                
                # Save to session if enabled
                if self.session_manager:
                    await self.session_manager.add_agent_message_to_current_session(tool_message)
                
                # Replace tool call with result in response
                tool_call_text = f"TOOL_CALL: {tool_name}({params_str})"
                if result.success:
                    replacement = f"[Tool executed: {tool_name} - {result.output}]"
                else:
                    replacement = f"[Tool error: {tool_name} - {result.error}]"
                
                modified_response = modified_response.replace(tool_call_text, replacement)
                
                logger.info(f"Executed tool {tool_name}: success={result.success}")
                
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                tool_call_text = f"TOOL_CALL: {tool_name}({params_str})"
                replacement = f"[Tool error: {tool_name} - {str(e)}]"
                modified_response = modified_response.replace(tool_call_text, replacement)
        
        return modified_response
    
    def _parse_tool_params(self, params_str: str) -> dict[str, Any]:
        """Parse tool parameters from string format."""
        params = {}
        if not params_str.strip():
            return params
        
        # Simple parameter parsing - handles key="value" format
        param_pattern = r'(\w+)=(["\'])(.*?)\2'
        matches = re.findall(param_pattern, params_str)
        
        for key, quote, value in matches:
            # Try to convert to appropriate type
            if value.lower() in ('true', 'false'):
                params[key] = value.lower() == 'true'
            elif value.isdigit():
                params[key] = int(value)
            else:
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value
        
        return params
    
    def get_available_tools(self) -> dict[str, Any]:
        """Get information about available tools."""
        if not self._tools_initialized:
            return {}
        
        tools_info = {}
        for tool_name, tool in self.tool_registry.get_all().items():
            tools_info[tool_name] = {
                "description": tool.description,
                "category": tool.metadata.category.value,
                "parameters": [
                    {
                        "name": p.name,
                        "type": p.type,
                        "description": p.description,
                        "required": p.required,
                        "default": p.default
                    }
                    for p in tool.parameters
                ]
            }
        
        return tools_info
    
    async def get_current_session(self):
        """Get the current session if session management is enabled."""
        if self.session_manager:
            return await self.session_manager.get_current_session()
        return None
    
    async def start_new_session(self, title: Optional[str] = None):
        """Start a new session if session management is enabled."""
        if self.session_manager:
            return await self.session_manager.start_new_session(title)
        return None
    
    async def complete_current_session(self):
        """Complete the current session if session management is enabled."""
        if self.session_manager:
            return await self.session_manager.complete_current_session()
        return None
    
    async def get_session_statistics(self) -> dict:
        """Get session statistics if session management is enabled."""
        if self.session_manager:
            return await self.session_manager.get_session_statistics()
        return {"session_management": "disabled"}
    
    async def health_check(self) -> dict[str, bool]:
        """Check the health of all agent components."""
        health_status = {
            "llm": False,
            "stt": False,
            "overall": False
        }
        
        try:
            # Check LLM service
            health_status["llm"] = await self.llm_service.health_check()
            
            # STT doesn't have a direct health check, assume ok if initialized
            health_status["stt"] = self.stt_processor._initialized
            
            # Overall health
            health_status["overall"] = health_status["llm"] and health_status["stt"]
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            
        return health_status
    
    async def cleanup(self) -> None:
        """Clean up agent resources."""
        logger.info("Cleaning up voice agent...")
        
        try:
            await self.stt_processor.cleanup()
            # LLM service doesn't require explicit cleanup
            
            # Clean up session manager
            if self.session_manager:
                await self.session_manager.close()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        self._set_state(AgentState.IDLE)
        logger.info("Voice agent cleanup complete")
    
    async def _single_segment(self, segment):
        """Helper to yield a single audio segment."""
        yield segment