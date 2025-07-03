#!/usr/bin/env python3
"""
Test script for session management system.
"""

import asyncio
import logging
from datetime import datetime
from uuid import uuid4

from transcriber.config import Settings
from transcriber.session.manager import SessionManager
from transcriber.session.models import Session, SessionMessage, MessageType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_session_storage():
    """Test basic session storage functionality."""
    logger.info("Testing session storage...")
    
    # Create session manager
    settings = Settings()
    session_manager = SessionManager(data_dir="./test_data")
    
    try:
        # Initialize
        await session_manager.initialize()
        
        # Start new session
        session = await session_manager.start_new_session("Test Session")
        logger.info(f"Created session: {session.id} - {session.title}")
        
        # Add some messages
        await session_manager.add_message_to_current_session(
            "Hello, this is a test message",
            MessageType.USER
        )
        
        await session_manager.add_message_to_current_session(
            "Hello! I'm here to help you.",
            MessageType.ASSISTANT
        )
        
        await session_manager.add_message_to_current_session(
            "Tool executed successfully",
            MessageType.TOOL,
            metadata={"tool_name": "test_tool", "success": True}
        )
        
        # Get current session
        current = await session_manager.get_current_session()
        logger.info(f"Current session has {len(current.messages)} messages")
        
        # Complete session
        completed = await session_manager.complete_current_session()
        logger.info(f"Completed session: {completed.status}")
        
        # List sessions
        sessions = await session_manager.list_sessions(limit=10)
        logger.info(f"Found {len(sessions)} sessions")
        
        # Get statistics
        stats = await session_manager.get_session_statistics()
        logger.info(f"Session statistics: {stats}")
        
        # Export session
        export_path = await session_manager.export_session(
            completed.id, 
            format="json"
        )
        logger.info(f"Exported session to: {export_path}")
        
        # Export as text
        text_path = await session_manager.export_session(
            completed.id,
            format="txt"
        )
        logger.info(f"Exported session as text to: {text_path}")
        
        logger.info("✅ Session storage test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Session storage test failed: {e}")
        raise
    
    finally:
        await session_manager.close()


async def test_agent_integration():
    """Test session integration with agent."""
    logger.info("Testing agent integration...")
    
    try:
        from transcriber.agent.core import VoiceAgent
        
        # Create settings with session enabled
        settings = Settings()
        settings.session.enabled = True
        settings.session.auto_start_session = True
        settings.data_dir = "./test_data"
        
        # Create agent
        agent = VoiceAgent(settings)
        await agent.initialize()
        
        # Test text input (should create session automatically)
        response = await agent.process_text_input("Hello, can you help me?")
        logger.info(f"Agent response: {response}")
        
        # Check session was created
        current_session = await agent.get_current_session()
        if current_session:
            logger.info(f"Session created: {current_session.id}")
            logger.info(f"Messages in session: {len(current_session.messages)}")
        else:
            logger.warning("No current session found")
        
        # Get session statistics
        stats = await agent.get_session_statistics()
        logger.info(f"Agent session stats: {stats}")
        
        # Test another message
        response2 = await agent.process_text_input("What's 2 + 2?")
        logger.info(f"Second response: {response2}")
        
        # Complete session
        completed = await agent.complete_current_session()
        if completed:
            logger.info(f"Session completed: {completed.id}")
        
        logger.info("✅ Agent integration test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Agent integration test failed: {e}")
        raise
    
    finally:
        await agent.cleanup()


async def main():
    """Run all tests."""
    logger.info("Starting session management tests...")
    
    try:
        await test_session_storage()
        await test_agent_integration()
        logger.info("🎉 All tests passed!")
        
    except Exception as e:
        logger.error(f"💥 Tests failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)