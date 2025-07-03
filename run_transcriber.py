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
    console.print("  [cyan]query[/cyan]       - Send a single query with persistent memory")
    console.print("  [cyan]memory-stats[/cyan] - Show memory system statistics")
    console.print("  [cyan]memory-list[/cyan] - List stored memories")
    console.print("  [cyan]devices[/cyan]     - List available audio devices")
    console.print("  [cyan]tools[/cyan]       - List available AI tools")
    console.print("  [cyan]help[/cyan]        - Show this help message")
    
    console.print("\n[bold]Examples:[/bold]")
    console.print("  [dim]poetry run transcriber demo[/dim]")
    console.print("  [dim]poetry run transcriber voice[/dim]           # Real microphone")
    console.print("  [dim]poetry run transcriber voice 5[/dim]         # Use device ID 5")
    console.print("  [dim]poetry run transcriber voice-demo[/dim]      # Simulated demo")
    console.print("  [dim]poetry run transcriber chat[/dim]")
    console.print("  [dim]poetry run transcriber query \"Hello\"[/dim]       # Single query")
    console.print("  [dim]poetry run transcriber memory-stats[/dim]    # Memory info")
    console.print("  [dim]poetry run transcriber memory-list 5[/dim]   # Show 5 memories")

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


async def run_query():
    """Run a single query with memory."""
    import os
    from transcriber.agent.query_agent import QueryAgent
    from transcriber.config import settings
    from rich.panel import Panel
    from rich.syntax import Syntax
    import json
    
    if len(sys.argv) < 3:
        console.print("[red]Error: Query message required[/red]")
        console.print("Usage: transcriber query \"your message here\"")
        return
    
    query_message = sys.argv[2]
    
    # Set environment variable for ChromaDB compatibility
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    
    try:
        console.print(f"[cyan]Processing query:[/cyan] {query_message}")
        console.print(f"[cyan]Memory enabled:[/cyan] {settings.memory.enabled}")
        
        # Create and initialize query agent
        query_agent = QueryAgent(settings)
        await query_agent.initialize()
        
        try:
            # Process the query
            result = await query_agent.process_query(
                query=query_message,
                use_memory=True,
                store_interaction=True,
                verbose=True
            )
            
            # Display results
            if result.error:
                console.print(f"[red]Error:[/red] {result.error}")
                return
            
            # Show memory context if available
            if (result.memory_context and
                    result.memory_context.has_relevant_context()):
                context_panel = Panel(
                    result.memory_context.get_context_text(),
                    title="[yellow]Memory Context Used[/yellow]",
                    border_style="yellow"
                )
                console.print(context_panel)
                console.print()
            
            # Show main response
            response_panel = Panel(
                result.response,
                title="[green]Agent Response[/green]",
                border_style="green"
            )
            console.print(response_panel)
            
            # Show metadata
            metadata = {
                "processing_time": f"{result.processing_time:.2f}s",
                "memory_context_used": result.memory_context is not None,
                "relevant_memories": (
                    len(result.memory_context.relevant_memories)
                    if result.memory_context else 0
                )
            }
            
            metadata_text = json.dumps(metadata, indent=2)
            metadata_panel = Panel(
                Syntax(metadata_text, "json", theme="monokai"),
                title="[blue]Processing Metadata[/blue]",
                border_style="blue"
            )
            console.print(metadata_panel)
        
        finally:
            await query_agent.cleanup()
    
    except Exception as e:
        console.print(f"[red]Query failed:[/red] {e}")


async def show_memory_stats():
    """Show memory system statistics."""
    import os
    from transcriber.memory.manager import MemoryManager
    from transcriber.config import settings
    
    if not settings.memory.enabled:
        console.print("[yellow]Memory system is disabled.[/yellow]")
        return
    
    # Set environment variable for ChromaDB compatibility
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    
    memory_manager = MemoryManager(settings.memory)
    
    try:
        # Get statistics using the manager's method
        stats = await memory_manager.get_memory_statistics()
        
        console.print("[bold]Memory System Statistics[/bold]")
        
        if stats.get("error"):
            console.print(f"[red]Error:[/red] {stats['error']}")
            return
        
        console.print(f"[cyan]Enabled:[/cyan] {stats.get('enabled', False)}")
        console.print(f"[cyan]Total memories:[/cyan] {stats.get('total_memories', 0)}")
        console.print(f"[cyan]Embedding model:[/cyan] {stats.get('embedding_model', 'N/A')}")
        console.print(f"[cyan]Embedding strategy:[/cyan] {stats.get('embedding_strategy', 'N/A')}")
        console.print(f"[cyan]Cache size:[/cyan] {stats.get('cache_size', 0)}")
        console.print(f"[cyan]Storage path:[/cyan] {stats.get('storage_path', 'N/A')}")
        
    except Exception as e:
        console.print(f"[red]Error getting memory stats: {e}[/red]")
    finally:
        await memory_manager.close()


async def list_memories():
    """List stored memories."""
    import os
    from transcriber.memory.manager import MemoryManager
    from transcriber.config import settings
    
    if not settings.memory.enabled:
        console.print("[yellow]Memory system is disabled.[/yellow]")
        return
    
    # Set environment variable for ChromaDB compatibility
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    
    # Get limit from command line args
    limit = 10
    if len(sys.argv) > 2:
        try:
            limit = int(sys.argv[2])
        except ValueError:
            console.print("[yellow]Invalid limit, using default of 10[/yellow]")
    
    memory_manager = MemoryManager(settings.memory)
    
    try:
        await memory_manager.initialize()
        collection = memory_manager.storage.collection
        
        if not collection:
            console.print("[yellow]Memory collection not initialized.[/yellow]")
            return
            
        count = collection.count()
        
        if count == 0:
            console.print("[yellow]No memories stored yet.[/yellow]")
            return
        
        # Get memories with metadata
        results = collection.get(
            limit=min(limit, count),
            include=['documents', 'metadatas']
        )
        
        if not results or not results.get('documents'):
            console.print("[yellow]No memories found.[/yellow]")
            return
        
        docs = results['documents'] or []
        metas = results['metadatas'] or []
        
        console.print(f"[bold]Stored Memories[/bold] (showing {len(docs)} of {count})")
        console.print()
        
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            console.print(f"[cyan]Memory {i+1}:[/cyan]")
            content = doc[:100] + ('...' if len(doc) > 100 else '')
            console.print(f"  [green]Content:[/green] {content}")
            
            if meta:
                entry_type = meta.get('entry_type', 'unknown')
                console.print(f"  [blue]Type:[/blue] {entry_type}")
                timestamp = meta.get('timestamp', 'unknown')
                console.print(f"  [blue]Timestamp:[/blue] {timestamp}")
                if 'user_id' in meta:
                    console.print(f"  [blue]User ID:[/blue] {meta['user_id']}")
            
            console.print()
        
        if count > limit:
            console.print(f"[dim]... and {count - limit} more memories[/dim]")
    
    except Exception as e:
        console.print(f"[red]Error listing memories: {e}[/red]")
    finally:
        await memory_manager.close()

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
    elif command == "query":
        asyncio.run(run_query())
    elif command == "memory-stats":
        asyncio.run(show_memory_stats())
    elif command == "memory-list":
        asyncio.run(list_memories())
    elif command == "devices":
        list_devices()
    elif command in ["list-tools", "tools"]:
        list_tools()
    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        console.print("Run '[cyan]python run_transcriber.py help[/cyan]' for available commands.")

if __name__ == "__main__":
    main()