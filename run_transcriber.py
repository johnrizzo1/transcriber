#!/usr/bin/env python3
"""
Simple launcher for the transcriber CLI that handles the help properly.
"""

import sys
from rich.console import Console
from rich.panel import Panel

console = Console()

def show_help():
    """Show the transcriber help."""
    console.print(Panel.fit(
        "[bold green]Transcriber AI Voice Agent[/bold green]\n"
        "Voice interface for AI tool execution",
        border_style="green"
    ))
    
    console.print("\n[bold]Available Commands:[/bold]")
    console.print("  [cyan]demo[/cyan]        - Run text-only agent demo")
    console.print("  [cyan]chat[/cyan]        - Interactive text chat mode") 
    console.print("  [cyan]voice[/cyan]       - Real-time voice interface (with microphone)")
    console.print("  [cyan]voice-demo[/cyan]  - Voice pipeline demo (simulated inputs)")
    console.print("  [cyan]devices[/cyan]     - List available audio devices")
    console.print("  [cyan]tools[/cyan]       - List available AI tools")
    console.print("  [cyan]help[/cyan]        - Show this help message")
    
    console.print("\n[bold]Examples:[/bold]")
    console.print("  [dim]poetry run transcriber demo[/dim]")
    console.print("  [dim]poetry run transcriber voice[/dim]           # Real microphone")
    console.print("  [dim]poetry run transcriber voice 5[/dim]         # Use device ID 5")
    console.print("  [dim]poetry run transcriber voice-demo[/dim]      # Simulated demo")
    console.print("  [dim]poetry run transcriber chat[/dim]")

async def run_demo():
    """Run the text demo."""
    from transcriber.config import settings
    from transcriber.agent.text_agent import TextOnlyAgent
    from rich.console import Console
    
    console = Console()
    console.print("[bold green]🤖 Transcriber AI Voice Agent Demo[/bold green]")
    console.print("[dim]Text-only mode[/dim]\n")
    
    try:
        agent = TextOnlyAgent(settings)
        await agent.initialize()
        
        console.print("[green]✅ Agent initialized successfully![/green]\n")
        
        # Demo conversation
        questions = [
            "Hello! What are you?",
            "What's 2+2?", 
            "Can you write a haiku about programming?",
            "Thanks for the demo!"
        ]
        
        for i, question in enumerate(questions, 1):
            console.print(f"[bold blue]Q{i}: {question}[/bold blue]")
            console.print("[green]🤖 Agent:[/green] ", end="")
            
            # Stream the response
            async for chunk in agent.process_text_input_stream(question):
                console.print(chunk, end="", style="green")
            
            console.print("\n")
        
        await agent.cleanup()
        console.print("[green]✅ Demo completed successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Demo failed: {e}[/red]")

async def run_voice_demo():
    """Run the voice pipeline demo."""
    try:
        from transcriber.voice_interface import run_voice_interface
        from transcriber.config import settings
        
        # Check if user wants specific device
        device_id = None
        if len(sys.argv) > 2:
            try:
                device_id = int(sys.argv[2])
                console.print(f"[cyan]Using audio device ID: {device_id}[/cyan]")
            except ValueError:
                console.print("[yellow]Invalid device ID, using default[/yellow]")
        
        await run_voice_interface(settings, device_id)
    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        console.print("[yellow]Try running: poetry install[/yellow]")
        raise

async def run_voice_pipeline_demo():
    """Run the voice pipeline demo with simulated inputs."""
    try:
        from transcriber.simple_voice import run_simple_voice_demo
        from transcriber.config import settings
        await run_simple_voice_demo(settings)
    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        console.print("[yellow]Try running: poetry install[/yellow]")
        raise

async def run_chat():
    """Run interactive chat."""
    from transcriber.agent.text_agent import run_interactive_chat
    from transcriber.config import settings
    await run_interactive_chat(settings)

def list_devices():
    """List audio devices."""
    try:
        from transcriber.audio import list_audio_devices
        
        console.print("[bold]Available Audio Devices:[/bold]")
        devices = list_audio_devices()
        
        input_devices = [d for d in devices if d['is_input']]
        output_devices = [d for d in devices if d['is_output']]
        
        console.print("\n[cyan]🎤 Input devices (microphones):[/cyan]")
        for device in input_devices:
            console.print(f"  {device['id']}: {device['name']} ({device['max_input_channels']} channels)")
        
        console.print("\n[cyan]🔊 Output devices (speakers):[/cyan]")
        for device in output_devices:
            console.print(f"  {device['id']}: {device['name']} ({device['max_output_channels']} channels)")
            
    except Exception as e:
        console.print(f"[red]Error listing devices: {e}[/red]")

def list_tools():
    """List available AI tools."""
    try:
        from transcriber.tools.registry import discover_tools, get_registry
        
        console.print("[bold]🛠️  Discovering available tools...[/bold]")
        
        # Discover all tools
        tools = discover_tools()
        registry = get_registry()
        
        if not tools:
            console.print("[yellow]No tools found[/yellow]")
            return
        
        console.print(f"\n[green]Found {len(tools)} built-in tools:[/green]\n")
        
        # Group tools by category
        by_category = {}
        for tool_name in tools:
            tool = registry.get(tool_name)
            category = tool.metadata.category.value
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(tool)
        
        # Display tools by category
        for category, tool_list in sorted(by_category.items()):
            console.print(f'[bold cyan]📁 {category.upper()}[/bold cyan] ({len(tool_list)} tools)')
            for tool in sorted(tool_list, key=lambda x: x.name):
                # Show permissions if any
                perms = tool.metadata.permissions
                perm_str = f" [dim]({', '.join(p.value for p in perms)})[/dim]" if perms else ""
                console.print(f'   • [green]{tool.name:<20}[/green] - {tool.description}{perm_str}')
            console.print()
        
        console.print(f"[dim]Use 'transcriber tools info <tool_name>' for detailed information about a specific tool[/dim]")
            
    except Exception as e:
        console.print(f"[red]Error listing tools: {e}[/red]")

def main():
    """Main CLI entry point."""
    import asyncio
    
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command in ["help", "--help", "-h"]:
        show_help()
    elif command == "demo":
        asyncio.run(run_demo())
    elif command == "voice":
        asyncio.run(run_voice_demo())
    elif command == "voice-demo":
        asyncio.run(run_voice_pipeline_demo())
    elif command == "chat":
        asyncio.run(run_chat())
    elif command == "devices":
        list_devices()
    elif command in ["list-tools", "tools"]:
        list_tools()
    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        console.print("Run '[cyan]python run_transcriber.py help[/cyan]' for available commands.")

if __name__ == "__main__":
    main()