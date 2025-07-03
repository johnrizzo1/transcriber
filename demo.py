#!/usr/bin/env python3
"""
Demo script to show the transcriber agent working.
"""

import asyncio
from rich.console import Console
from transcriber.config import settings
from transcriber.agent.text_agent import TextOnlyAgent

console = Console()

async def demo():
    """Demo the text-only agent."""
    console.print("[bold green]🤖 Transcriber AI Voice Agent Demo[/bold green]")
    console.print("[dim]Text-only mode (speech dependencies not installed)[/dim]\n")
    
    try:
        agent = TextOnlyAgent(settings)
        await agent.initialize()
        
        console.print("[green]✅ Agent initialized successfully![/green]\n")
        
        # Demo conversation
        questions = [
            "Hello! What are you?",
            "What's the weather like on Mars?", 
            "Can you write a haiku about coffee?",
            "What are the three laws of robotics?"
        ]
        
        for i, question in enumerate(questions, 1):
            console.print(f"[bold blue]Q{i}: {question}[/bold blue]")
            console.print("[green]🤖 Agent:[/green] ", end="")
            
            # Stream the response
            async for chunk in agent.process_text_input_stream(question):
                console.print(chunk, end="", style="green")
            
            console.print("\n")
        
        # Show conversation stats
        history = agent.get_conversation_history()
        console.print(f"[dim]💬 Total conversation: {len(history)} messages[/dim]")
        
        await agent.cleanup()
        console.print("\n[green]✅ Demo completed successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Demo failed: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(demo())