#!/usr/bin/env python3
"""
Main entry point for the Transcriber AI Voice Agent.
"""

import asyncio
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(
    name="transcriber",
    help="AI Voice Agent - Natural voice interface for AI tool execution",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.command()
def start(
    model: str = typer.Option(
        "llama3.2:3b", "--model", "-m", help="Ollama model to use"
    ),
    device: Optional[int] = typer.Option(
        None, "--device", "-d", help="Audio input device index"
    ),
    list_devices: bool = typer.Option(
        False, "--list-devices", "-l", help="List available audio devices"
    ),
):
    """Start the AI voice agent."""
    from .audio import list_audio_devices
    from .config import settings
    from .utils import setup_logging
    
    # Setup logging
    logger = setup_logging(settings.log_level, debug=settings.debug)
    
    console.print(
        Panel.fit(
            "[bold green]Transcriber AI Voice Agent[/bold green]\n"
            "Voice interface for AI tool execution",
            border_style="green",
        )
    )
    
    if list_devices:
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
        
        return
    
    console.print(f"[cyan]Using model:[/cyan] {model}")
    
    if device is not None:
        settings.audio.input_device = device
        console.print(f"[cyan]Using input device:[/cyan] {device}")
    
    # Start the voice agent
    asyncio.run(run_voice_agent(model))


@app.command()
def chat(
    model: str = typer.Option(
        "llama3.2:3b", "--model", "-m", help="Ollama model to use"
    ),
):
    """Start text-only chat mode for testing."""
    from .config import settings
    from .pipeline import TextModeAgent
    
    # Update model setting
    settings.agent.model = model
    
    console.print("[yellow]Starting text-only chat mode...[/yellow]")
    
    async def run_text_chat():
        text_agent = TextModeAgent(settings)
        await text_agent.initialize()
        await text_agent.run_text_mode()
    
    asyncio.run(run_text_chat())


@app.command()
def list_tools(
    category: Optional[str] = typer.Option(
        None, "--category", "-c", help="Filter by tool category"
    ),
    search: Optional[str] = typer.Option(
        None, "--search", "-s", help="Search tools by name or description"
    ),
    detailed: bool = typer.Option(
        False, "--detailed", "-d", help="Show detailed tool information"
    ),
):
    """List all available tools with descriptions and usage examples."""
    from .tools import get_registry, discover_tools
    from .tools.base import ToolCategory
    from rich.table import Table
    from rich.panel import Panel
    
    # Initialize tools if not already done
    try:
        discovered = discover_tools()
        if discovered:
            console.print(f"[green]Discovered {len(discovered)} tools[/green]")
    except Exception as e:
        console.print(f"[yellow]Warning: Tool discovery failed: {e}[/yellow]")
    
    registry = get_registry()
    tools = registry.get_all()
    
    if not tools:
        console.print(
            "[red]No tools available. Run tool discovery first.[/red]"
        )
        return
    
    # Filter by category if specified
    if category:
        try:
            cat_enum = ToolCategory(category.lower())
            tools = {
                name: tool for name, tool in tools.items()
                if tool.metadata.category == cat_enum
            }
        except ValueError:
            console.print(f"[red]Invalid category: {category}[/red]")
            categories_str = ', '.join([c.value for c in ToolCategory])
            console.print(f"[cyan]Available categories:[/cyan] {categories_str}")
            return
    
    # Filter by search term if specified
    if search:
        search_lower = search.lower()
        tools = {
            name: tool for name, tool in tools.items()
            if (search_lower in name.lower() or
                search_lower in tool.description.lower())
        }
    
    if not tools:
        console.print(
            "[yellow]No tools match the specified criteria.[/yellow]"
        )
        return
    
    if detailed:
        # Show detailed information for each tool
        for name, tool in sorted(tools.items()):
            _show_detailed_tool_info(tool, console)
    else:
        # Show summary table
        _show_tools_table(tools, console)
    
    # Show summary statistics
    categories = registry.list_categories()
    console.print(f"\n[cyan]Summary:[/cyan] {len(tools)} tools shown")
    if not category and not search:
        categories_list = ', '.join(categories.keys())
        console.print(f"[cyan]Categories:[/cyan] {categories_list}")


def _show_tools_table(tools: dict, console):
    """Show tools in a formatted table."""
    from rich.table import Table
    
    table = Table(
        title="Available Tools",
        show_header=True,
        header_style="bold magenta"
    )
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Category", style="green")
    table.add_column("Description", style="white")
    table.add_column("Permissions", style="yellow")
    
    for name, tool in sorted(tools.items()):
        permissions = (
            ", ".join([p.value for p in tool.metadata.permissions]) or "None"
        )
        description = (
            tool.description[:60] + "..."
            if len(tool.description) > 60
            else tool.description
        )
        table.add_row(name, tool.metadata.category.value, description, permissions)
    
    console.print(table)


def _show_detailed_tool_info(tool, console):
    """Show detailed information for a single tool."""
    # Main info panel
    permissions_str = (
        ', '.join([p.value for p in tool.metadata.permissions]) or 'None'
    )
    info_lines = [
        f"[bold cyan]{tool.name}[/bold cyan]",
        f"[green]Category:[/green] {tool.metadata.category.value}",
        f"[green]Version:[/green] {tool.metadata.version}",
        f"[green]Author:[/green] {tool.metadata.author}",
        f"[green]Permissions:[/green] {permissions_str}",
        "",
        "[yellow]Description:[/yellow]",
        tool.description
    ]
    
    # Parameters
    if tool.parameters:
        info_lines.extend(["", "[yellow]Parameters:[/yellow]"])
        for param in tool.parameters:
            required = "[red]*[/red]" if param.required else ""
            default = (
                f" (default: {param.default})"
                if param.default is not None else ""
            )
            choices = (
                f" [choices: {', '.join(param.choices)}]"
                if param.choices else ""
            )
            param_line = (
                f"  • [cyan]{param.name}[/cyan]{required} ({param.type}): "
                f"{param.description}{default}{choices}"
            )
            info_lines.append(param_line)
    
    # Examples
    if tool.metadata.examples:
        info_lines.extend(["", "[yellow]Examples:[/yellow]"])
        for example in tool.metadata.examples:
            info_lines.append(f"  [dim]{example}[/dim]")
    
    console.print(Panel("\n".join(info_lines), border_style="blue"))
    console.print()


@app.command()
def list_sessions(
    limit: int = typer.Option(
        20, "--limit", "-l", help="Maximum number of sessions to show"
    ),
    status: Optional[str] = typer.Option(
        None, "--status", "-s", help="Filter by session status"
    ),
    search: Optional[str] = typer.Option(
        None, "--search", help="Search sessions by title or content"
    ),
    sort_by: str = typer.Option(
        "created_at", "--sort", help="Sort by field"
    ),
    reverse: bool = typer.Option(
        True, "--reverse/--no-reverse", help="Sort in reverse order"
    ),
):
    """List stored conversation sessions with filtering and sorting options."""
    import asyncio
    from .session.manager import SessionManager
    from .session.models import SessionStatus
    from rich.table import Table
    from datetime import datetime
    
    async def list_sessions_async():
        manager = SessionManager()
        await manager.initialize()
        
        try:
            # Get sessions with filtering
            if search:
                sessions = await manager.search_sessions(search, limit)
            else:
                status_filter = None
                if status:
                    try:
                        status_filter = SessionStatus(status.lower())
                    except ValueError:
                        console.print(f"[red]Invalid status: {status}[/red]")
                        valid_statuses = [s.value for s in SessionStatus]
                        console.print(f"[cyan]Valid statuses:[/cyan] {', '.join(valid_statuses)}")
                        return
                
                sessions = await manager.list_sessions(
                    limit=limit,
                    status=status_filter
                )
            
            if not sessions:
                console.print("[yellow]No sessions found.[/yellow]")
                return
            
            # Create table
            table = Table(
                title=f"Sessions ({len(sessions)} found)",
                show_header=True,
                header_style="bold magenta"
            )
            table.add_column("ID", style="cyan", no_wrap=True, width=8)
            table.add_column("Title", style="white", width=30)
            table.add_column("Status", style="green", width=10)
            table.add_column("Created", style="blue", width=16)
            table.add_column("Messages", style="yellow", width=8)
            table.add_column("Duration", style="dim", width=10)
            
            for session in sessions:
                # Format creation time
                created_str = session.created_at.strftime("%m/%d %H:%M")
                
                # Calculate duration
                if session.completed_at:
                    duration = session.completed_at - session.created_at
                    duration_str = _format_duration(duration.total_seconds())
                else:
                    duration_str = "Active"
                
                # Truncate title if too long
                title = (
                    session.title[:27] + "..."
                    if len(session.title) > 30
                    else session.title
                )
                
                table.add_row(
                    str(session.id)[:8],
                    title,
                    session.status.value,
                    created_str,
                    str(session.metadata.total_messages),
                    duration_str
                )
            
            console.print(table)
            
            # Show statistics
            stats = await manager.get_session_statistics()
            console.print(f"\n[cyan]Total sessions:[/cyan] {stats['total_sessions']}")
            console.print(f"[cyan]Active sessions:[/cyan] {stats['active_sessions']}")
            console.print(f"[cyan]Completed sessions:[/cyan] {stats['completed_sessions']}")
            
        finally:
            await manager.close()
    
    asyncio.run(list_sessions_async())


def _format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes}m"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h{minutes}m"


@app.command()
def replay(
    session_id: str = typer.Argument(
        "latest", help="Session ID to replay (latest if not specified)"
    ),
    format: str = typer.Option(
        "rich", "--format", "-f", help="Output format (rich, plain, json)"
    ),
    show_timestamps: bool = typer.Option(
        True, "--timestamps/--no-timestamps", help="Show message timestamps"
    ),
):
    """Replay a previous conversation session."""
    import asyncio
    from .session.manager import SessionManager
    from uuid import UUID
    from rich.panel import Panel
    from rich.syntax import Syntax
    import json
    
    async def replay_session_async():
        manager = SessionManager()
        await manager.initialize()
        
        try:
            # Handle "latest" session
            if session_id == "latest":
                sessions = await manager.list_sessions(limit=1)
                if not sessions:
                    console.print("[red]No sessions found.[/red]")
                    return
                target_session = sessions[0]
            else:
                # Try to parse as UUID
                try:
                    if len(session_id) == 8:
                        # Short ID - find matching session
                        sessions = await manager.list_sessions(limit=100)
                        matching = [s for s in sessions if str(s.id).startswith(session_id)]
                        if not matching:
                            console.print(f"[red]No session found with ID starting with: {session_id}[/red]")
                            return
                        elif len(matching) > 1:
                            console.print(f"[yellow]Multiple sessions match '{session_id}':[/yellow]")
                            for s in matching[:5]:
                                console.print(f"  {str(s.id)[:8]} - {s.title}")
                            return
                        target_session = matching[0]
                    else:
                        session_uuid = UUID(session_id)
                        target_session = await manager.get_session(session_uuid)
                        if not target_session:
                            console.print(f"[red]Session not found: {session_id}[/red]")
                            return
                except ValueError:
                    console.print(f"[red]Invalid session ID: {session_id}[/red]")
                    return
            
            # Get messages for replay
            messages = await manager.replay_session(target_session.id)
            
            if format == "json":
                # JSON output
                output = {
                    "session": target_session.to_dict(),
                    "messages": [msg.to_dict() for msg in messages]
                }
                console.print(json.dumps(output, indent=2, default=str))
                return
            
            # Show session header
            header_lines = [
                f"[bold cyan]Session: {target_session.title}[/bold cyan]",
                f"[green]ID:[/green] {target_session.id}",
                f"[green]Created:[/green] {target_session.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
                f"[green]Status:[/green] {target_session.status.value}",
                f"[green]Messages:[/green] {len(messages)}"
            ]
            
            if target_session.completed_at:
                duration = target_session.completed_at - target_session.created_at
                header_lines.append(f"[green]Duration:[/green] {_format_duration(duration.total_seconds())}")
            
            console.print(Panel("\n".join(header_lines), border_style="blue"))
            console.print()
            
            # Show messages
            for msg in messages:
                if format == "plain":
                    timestamp = f"[{msg.timestamp.strftime('%H:%M:%S')}] " if show_timestamps else ""
                    console.print(f"{timestamp}{msg.message_type.value.title()}: {msg.content}")
                else:
                    # Rich format
                    timestamp = msg.timestamp.strftime('%H:%M:%S') if show_timestamps else ""
                    
                    if msg.message_type.value == "user":
                        style = "bold blue"
                        icon = "🗣️"
                    elif msg.message_type.value == "assistant":
                        style = "bold green"
                        icon = "🤖"
                    elif msg.message_type.value == "tool":
                        style = "bold yellow"
                        icon = "🔧"
                    else:
                        style = "dim"
                        icon = "ℹ️"
                    
                    header = f"{icon} [{style}]{msg.message_type.value.title()}[/{style}]"
                    if show_timestamps:
                        header += f" [dim]({timestamp})[/dim]"
                    
                    console.print(header)
                    console.print(f"  {msg.content}")
                    console.print()
            
        finally:
            await manager.close()
    
    asyncio.run(replay_session_async())


@app.command()
def export(
    session_id: str = typer.Argument(
        "latest", help="Session ID to export (latest if not specified)"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path (auto-generated if not specified)"
    ),
    format: str = typer.Option(
        "json", "--format", "-f", help="Export format (json, txt)"
    ),
    include_metadata: bool = typer.Option(
        True, "--metadata/--no-metadata", help="Include session metadata"
    ),
):
    """Export a conversation session to file."""
    import asyncio
    from .session.manager import SessionManager
    from uuid import UUID
    import json
    
    async def export_session_async():
        manager = SessionManager()
        await manager.initialize()
        
        try:
            # Handle "latest" session
            if session_id == "latest":
                sessions = await manager.list_sessions(limit=1)
                if not sessions:
                    console.print("[red]No sessions found.[/red]")
                    return
                target_session = sessions[0]
            else:
                # Try to parse as UUID
                try:
                    if len(session_id) == 8:
                        # Short ID - find matching session
                        sessions = await manager.list_sessions(limit=100)
                        matching = [s for s in sessions if str(s.id).startswith(session_id)]
                        if not matching:
                            console.print(f"[red]No session found with ID starting with: {session_id}[/red]")
                            return
                        elif len(matching) > 1:
                            console.print(f"[yellow]Multiple sessions match '{session_id}':[/yellow]")
                            for s in matching[:5]:
                                console.print(f"  {str(s.id)[:8]} - {s.title}")
                            return
                        target_session = matching[0]
                    else:
                        session_uuid = UUID(session_id)
                        target_session = await manager.get_session(session_uuid)
                        if not target_session:
                            console.print(f"[red]Session not found: {session_id}[/red]")
                            return
                except ValueError:
                    console.print(f"[red]Invalid session ID: {session_id}[/red]")
                    return
            
            # Validate format
            if format not in ["json", "txt"]:
                console.print(f"[red]Unsupported format: {format}[/red]")
                console.print("[cyan]Supported formats:[/cyan] json, txt")
                return
            
            # Generate output path if not specified
            if output is None:
                timestamp = target_session.created_at.strftime("%Y%m%d_%H%M%S")
                filename = f"session_{timestamp}.{format}"
                output_path = Path("./exports") / filename
            else:
                output_path = output
                # Add extension if not present
                if not output_path.suffix:
                    output_path = output_path.with_suffix(f".{format}")
            
            # Export the session
            try:
                exported_path = await manager.export_session(
                    target_session.id,
                    format=format,
                    output_path=str(output_path)
                )
                
                console.print(f"[green]✓ Session exported successfully![/green]")
                console.print(f"[cyan]File:[/cyan] {exported_path}")
                console.print(f"[cyan]Format:[/cyan] {format.upper()}")
                console.print(f"[cyan]Size:[/cyan] {Path(exported_path).stat().st_size} bytes")
                
                # Show preview for text format
                if format == "txt" and Path(exported_path).stat().st_size < 2000:
                    console.print("\n[yellow]Preview:[/yellow]")
                    with open(exported_path, 'r', encoding='utf-8') as f:
                        preview = f.read()[:500]
                        console.print(f"[dim]{preview}{'...' if len(preview) == 500 else ''}[/dim]")
                
            except Exception as e:
                console.print(f"[red]Export failed: {e}[/red]")
                return
            
        finally:
            await manager.close()
    
    asyncio.run(export_session_async())


@app.command()
def configure(
    interactive: bool = typer.Option(
        True, "--interactive/--non-interactive", help="Interactive configuration mode"
    ),
    show_current: bool = typer.Option(
        False, "--show", help="Show current configuration"
    ),
    set_values: List[str] = typer.Option(
        [], "--set", help="Set configuration values (format: key=value)"
    ),
):
    """Interactive configuration setup and management."""
    from .config import settings
    from rich.prompt import Prompt, Confirm
    from rich.panel import Panel
    import os
    
    if show_current:
        # Show current configuration
        config_info = [
            "[bold cyan]Current Configuration[/bold cyan]",
            "",
            f"[green]Agent Model:[/green] {settings.agent.model}",
            f"[green]STT Model:[/green] {settings.whisper.model}",
            f"[green]Edge TTS Voice:[/green] {settings.edge_voice}",
            f"[green]Audio Sample Rate:[/green] {settings.audio.sample_rate}Hz",
            f"[green]VAD Threshold:[/green] {settings.voice.vad_threshold}",
            f"[green]Log Level:[/green] {settings.log_level}",
            f"[green]Debug Mode:[/green] {settings.debug}",
            "",
            "[yellow]Environment Variables:[/yellow]",
        ]
        
        # Show relevant environment variables
        env_vars = [
            "TRANSCRIBER_AGENT__MODEL",
            "TRANSCRIBER_WHISPER__MODEL",
            "TRANSCRIBER_EDGE_VOICE",
            "TRANSCRIBER_AUDIO__SAMPLE_RATE",
            "TRANSCRIBER_VOICE__VAD_THRESHOLD",
            "TRANSCRIBER_LOG_LEVEL"
        ]
        
        for var in env_vars:
            value = os.environ.get(var, "[dim]not set[/dim]")
            config_info.append(f"  {var}: {value}")
        
        console.print(Panel("\n".join(config_info), border_style="blue"))
        return
    
    # Handle --set options for non-interactive configuration
    if set_values:
        from .config import settings
        import os
        
        console.print("[bold cyan]Setting configuration values...[/bold cyan]")
        
        for setting in set_values:
            if '=' not in setting:
                console.print(f"[red]Error:[/red] Invalid format '{setting}'. Use key=value")
                continue
                
            key, value = setting.split('=', 1)
            env_key = f"TRANSCRIBER_{key.upper().replace('.', '__')}"
            
            # Set environment variable
            os.environ[env_key] = value
            console.print(f"[green]✓[/green] Set {key} = {value}")
        
        console.print("\n[yellow]Note:[/yellow] Changes are applied for this session only.")
        console.print("To make permanent changes, set environment variables or use interactive mode.")
        return
    
    if not interactive:
        console.print("[yellow]Use --set key=value to configure non-interactively[/yellow]")
        console.print("Use --show to view current configuration")
        return
    
    console.print(Panel(
        "[bold cyan]Transcriber Configuration Setup[/bold cyan]\n\n"
        "This will help you configure the AI Voice Agent.\n"
        "Press Enter to keep current values.",
        border_style="green"
    ))
    
    # Agent configuration
    console.print("\n[bold yellow]🤖 Agent Configuration[/bold yellow]")
    
    current_model = settings.agent.model
    new_model = Prompt.ask(
        f"LLM Model (current: {current_model})",
        default=current_model
    )
    
    # Audio configuration
    console.print("\n[bold yellow]🎵 Audio Configuration[/bold yellow]")
    
    current_sample_rate = settings.audio.sample_rate
    new_sample_rate = Prompt.ask(
        f"Sample Rate (current: {current_sample_rate})",
        default=str(current_sample_rate)
    )
    
    current_vad_threshold = settings.voice.vad_threshold
    new_vad_threshold = Prompt.ask(
        f"VAD Threshold (current: {current_vad_threshold})",
        default=str(current_vad_threshold)
    )
    
    # STT configuration
    console.print("\n[bold yellow]🎤 Speech-to-Text Configuration[/bold yellow]")
    
    current_stt_model = settings.whisper.model
    stt_models = ["tiny", "base", "small", "medium", "large"]
    console.print(f"Available models: {', '.join(stt_models)}")
    new_stt_model = Prompt.ask(
        f"Whisper Model (current: {current_stt_model})",
        default=current_stt_model
    )
    
    # TTS configuration
    console.print("\n[bold yellow]🔊 Text-to-Speech Configuration[/bold yellow]")
    
    current_edge_voice = settings.edge_voice
    console.print("Available voices: en-US-AriaNeural, en-US-JennyNeural, en-US-GuyNeural")
    new_edge_voice = Prompt.ask(
        f"Edge TTS Voice (current: {current_edge_voice})",
        default=current_edge_voice
    )
    
    # Debug configuration
    console.print("\n[bold yellow]🐛 Debug Configuration[/bold yellow]")
    
    current_debug = settings.debug
    new_debug = Confirm.ask(
        f"Enable debug mode (current: {current_debug})",
        default=current_debug
    )
    
    current_log_level = settings.log_level
    log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
    console.print(f"Available levels: {', '.join(log_levels)}")
    new_log_level = Prompt.ask(
        f"Log Level (current: {current_log_level})",
        default=current_log_level
    )
    
    # Show summary and confirm
    console.print("\n[bold cyan]Configuration Summary:[/bold cyan]")
    changes = []
    
    if new_model != current_model:
        changes.append(f"Agent Model: {current_model} → {new_model}")
    if int(new_sample_rate) != current_sample_rate:
        changes.append(f"Sample Rate: {current_sample_rate} → {new_sample_rate}")
    if float(new_vad_threshold) != current_vad_threshold:
        changes.append(f"VAD Threshold: {current_vad_threshold} → {new_vad_threshold}")
    if new_stt_model != current_stt_model:
        changes.append(f"STT Model: {current_stt_model} → {new_stt_model}")
    if new_edge_voice != current_edge_voice:
        changes.append(f"Edge TTS Voice: {current_edge_voice} → {new_edge_voice}")
    if new_debug != current_debug:
        changes.append(f"Debug Mode: {current_debug} → {new_debug}")
    if new_log_level != current_log_level:
        changes.append(f"Log Level: {current_log_level} → {new_log_level}")
    
    if not changes:
        console.print("[yellow]No changes detected.[/yellow]")
        return
    
    for change in changes:
        console.print(f"  • {change}")
    
    if not Confirm.ask("\nSave these changes?"):
        console.print("[yellow]Configuration not saved.[/yellow]")
        return
    
    # Generate environment variable commands
    console.print("\n[bold green]✓ Configuration saved![/bold green]")
    console.print("\n[yellow]To make these changes permanent, add these environment variables:[/yellow]")
    
    env_commands = []
    if new_model != current_model:
        env_commands.append(f"export TRANSCRIBER_AGENT__MODEL='{new_model}'")
    if int(new_sample_rate) != current_sample_rate:
        env_commands.append(f"export TRANSCRIBER_AUDIO__SAMPLE_RATE={new_sample_rate}")
    if float(new_vad_threshold) != current_vad_threshold:
        env_commands.append(f"export TRANSCRIBER_VOICE__VAD_THRESHOLD={new_vad_threshold}")
    if new_stt_model != current_stt_model:
        env_commands.append(f"export TRANSCRIBER_WHISPER__MODEL='{new_stt_model}'")
    if new_edge_voice != current_edge_voice:
        env_commands.append(f"export TRANSCRIBER_EDGE_VOICE='{new_edge_voice}'")
    if new_debug != current_debug:
        env_commands.append(f"export TRANSCRIBER_DEBUG={'true' if new_debug else 'false'}")
    if new_log_level != current_log_level:
        env_commands.append(f"export TRANSCRIBER_LOG_LEVEL='{new_log_level}'")
    
    for cmd in env_commands:
        console.print(f"  {cmd}")
    
    console.print("\n[dim]Add these to your ~/.bashrc, ~/.zshrc, or .envrc file[/dim]")


@app.command()
def cleanup(
    days: int = typer.Option(
        30, "--days", "-d", help="Delete sessions older than this many days"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be deleted without actually deleting"
    ),
    confirm: bool = typer.Option(
        False, "--confirm", help="Skip confirmation prompt"
    ),
):
    """Clean up old sessions and perform database maintenance."""
    import asyncio
    from .session.manager import SessionManager
    from rich.prompt import Confirm
    
    async def cleanup_async():
        manager = SessionManager()
        await manager.initialize()
        
        try:
            # Get statistics before cleanup
            stats_before = await manager.get_session_statistics()
            
            console.print(f"[cyan]Current Statistics:[/cyan]")
            console.print(f"  Total sessions: {stats_before['total_sessions']}")
            console.print(f"  Active sessions: {stats_before['active_sessions']}")
            console.print(f"  Completed sessions: {stats_before['completed_sessions']}")
            
            if dry_run:
                console.print(f"\n[yellow]DRY RUN: Would delete sessions older than {days} days[/yellow]")
                # For dry run, we'd need to implement a preview function
                console.print("[dim]Dry run functionality would show sessions to be deleted[/dim]")
                return
            
            if not confirm:
                if not Confirm.ask(f"\nDelete sessions older than {days} days?"):
                    console.print("[yellow]Cleanup cancelled.[/yellow]")
                    return
            
            # Perform cleanup
            console.print(f"\n[yellow]Cleaning up sessions older than {days} days...[/yellow]")
            
            deleted_count = await manager.cleanup_old_sessions(days)
            
            if deleted_count > 0:
                console.print(f"[green]✓ Deleted {deleted_count} old sessions[/green]")
                
                # Show statistics after cleanup
                stats_after = await manager.get_session_statistics()
                console.print(f"\n[cyan]Updated Statistics:[/cyan]")
                console.print(f"  Total sessions: {stats_after['total_sessions']}")
                console.print(f"  Active sessions: {stats_after['active_sessions']}")
                console.print(f"  Completed sessions: {stats_after['completed_sessions']}")
                
                freed_sessions = stats_before['total_sessions'] - stats_after['total_sessions']
                console.print(f"  [green]Freed: {freed_sessions} sessions[/green]")
            else:
                console.print(f"[yellow]No sessions older than {days} days found.[/yellow]")
            
        finally:
            await manager.close()
    
    asyncio.run(cleanup_async())


@app.command()
def benchmark(
    component: Optional[str] = typer.Option(
        None, "--component", "-c", help="Component to benchmark (audio, stt, llm, tts, pipeline)"
    ),
    iterations: int = typer.Option(
        100, "--iterations", "-i", help="Number of benchmark iterations"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file for benchmark results"
    ),
    format: str = typer.Option(
        "json", "--format", "-f", help="Output format (json, csv)"
    ),
):
    """Run performance benchmarks on pipeline components."""
    import asyncio
    from .performance.benchmarks import BenchmarkSuite
    from .performance.monitor import ComponentType
    
    async def run_benchmarks():
        suite = BenchmarkSuite()
        
        console.print("[cyan]Starting performance benchmarks...[/cyan]")
        
        if component:
            # Benchmark specific component
            try:
                comp_type = ComponentType(component.lower())
                console.print(f"[yellow]Benchmarking {component} component...[/yellow]")
                
                # This would need to be implemented based on actual component interfaces
                console.print(f"[red]Component-specific benchmarking not yet implemented for {component}[/red]")
                
            except ValueError:
                console.print(f"[red]Invalid component: {component}[/red]")
                valid_components = [c.value for c in ComponentType]
                console.print(f"[cyan]Valid components:[/cyan] {', '.join(valid_components)}")
                return
        else:
            # Run all benchmarks
            console.print("[yellow]Running comprehensive benchmark suite...[/yellow]")
            console.print(f"[dim]This may take several minutes with {iterations} iterations[/dim]")
            
            # Placeholder for comprehensive benchmarks
            console.print("[red]Comprehensive benchmarking not yet implemented[/red]")
            console.print("[cyan]Use --component to benchmark specific components[/cyan]")
        
        # Export results if requested
        if output:
            suite.export_benchmark_report(output, format)
            console.print(f"[green]Benchmark results exported to {output}[/green]")
    
    asyncio.run(run_benchmarks())


@app.command()
def profile(
    duration: int = typer.Option(
        60, "--duration", "-d", help="Profiling duration in seconds"
    ),
    component: Optional[str] = typer.Option(
        None, "--component", "-c", help="Component to profile"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file for profiling results"
    ),
    memory: bool = typer.Option(
        False, "--memory", help="Enable memory profiling"
    ),
):
    """Profile pipeline performance to identify bottlenecks."""
    import asyncio
    from .performance.profiler import ProfilerManager
    from .performance.monitor import ComponentType
    
    async def run_profiling():
        profiler = ProfilerManager()
        
        console.print(f"[cyan]Starting performance profiling for {duration} seconds...[/cyan]")
        
        if component:
            try:
                comp_type = ComponentType(component.lower())
                console.print(f"[yellow]Profiling {component} component...[/yellow]")
                
                # This would need to be integrated with actual pipeline
                console.print(f"[red]Component-specific profiling not yet implemented for {component}[/red]")
                
            except ValueError:
                console.print(f"[red]Invalid component: {component}[/red]")
                valid_components = [c.value for c in ComponentType]
                console.print(f"[cyan]Valid components:[/cyan] {', '.join(valid_components)}")
                return
        else:
            console.print("[yellow]Profiling entire pipeline...[/yellow]")
            console.print("[red]Pipeline profiling not yet implemented[/red]")
        
        # Export results if requested
        if output:
            profiler.export_profile_report(output)
            console.print(f"[green]Profiling results exported to {output}[/green]")
    
    asyncio.run(run_profiling())


@app.command()
def query(
    message: str = typer.Argument(
        ..., help="Query message to send to the agent"
    ),
    model: str = typer.Option(
        "llama3.2:3b", "--model", "-m", help="Ollama model to use"
    ),
    memory: bool = typer.Option(
        True, "--memory/--no-memory", help="Enable memory context retrieval"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show memory context and metadata"
    ),
    store: bool = typer.Option(
        True, "--store/--no-store", help="Store this interaction in memory"
    ),
):
    """Send a single query to the agent with persistent memory."""
    import asyncio
    import json
    from .agent.query_agent import QueryAgent
    from .config import settings
    from rich.panel import Panel
    from rich.syntax import Syntax

    async def process_query_async():
        # Update model setting
        settings.agent.model = model

        # Create and initialize query agent
        query_agent = QueryAgent(settings)

        try:
            # Show processing indicator
            if verbose:
                console.print(
                    f"[cyan]Processing query with model:[/cyan] {model}"
                )
                console.print(f"[cyan]Memory enabled:[/cyan] {memory}")

            # Process the query
            result = await query_agent.process_query(
                query=message,
                use_memory=memory,
                store_interaction=store,
                verbose=verbose
            )

            # Display results
            if result.error:
                console.print(f"[red]Error:[/red] {result.error}")
                return

            # Show memory context if verbose and available
            if (verbose and result.memory_context and
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

            # Show metadata if verbose
            if verbose:
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

    # Run the async query processing
    asyncio.run(process_query_async())


@app.command()
def memory_stats():
    """Show memory system statistics and status."""
    import asyncio
    from .memory import MemoryManager
    from .config import settings
    from rich.table import Table

    async def show_memory_stats():
        if not settings.memory.enabled:
            console.print("[yellow]Memory system is disabled[/yellow]")
            return

        memory_manager = MemoryManager(settings.memory)

        try:
            stats = await memory_manager.get_memory_statistics()

            # Create statistics table
            table = Table(title="Memory System Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")

            table.add_row(
                "Status",
                "✅ Enabled" if stats.get("enabled") else "❌ Disabled"
            )
            table.add_row(
                "Total Memories",
                str(stats.get("total_memories", "Unknown"))
            )
            table.add_row(
                "Embedding Model",
                stats.get("embedding_model", "Unknown")
            )
            table.add_row(
                "Embedding Strategy",
                stats.get("embedding_strategy", "Unknown")
            )
            table.add_row(
                "Cache Size",
                str(stats.get("cache_size", "Unknown"))
            )
            table.add_row(
                "Storage Path",
                stats.get("storage_path", "Unknown")
            )

            console.print(table)

            if stats.get("error"):
                console.print(f"[red]Error:[/red] {stats['error']}")

        finally:
            await memory_manager.close()

    asyncio.run(show_memory_stats())


@app.command()
def memory_cleanup(
    days: int = typer.Option(
        30, "--days", "-d",
        help="Delete memories older than this many days"
    ),
    confirm: bool = typer.Option(
        False, "--confirm", "-y", help="Skip confirmation prompt"
    ),
):
    """Clean up old memories from the vector database."""
    import asyncio
    from .memory import MemoryManager
    from .config import settings

    async def cleanup_memories():
        if not settings.memory.enabled:
            console.print("[yellow]Memory system is disabled[/yellow]")
            return

        if not confirm:
            response = typer.confirm(
                f"This will delete all memories older than {days} days. "
                "Continue?"
            )
            if not response:
                console.print("Cleanup cancelled")
                return

        memory_manager = MemoryManager(settings.memory)

        try:
            console.print(
                f"[cyan]Cleaning up memories older than {days} days..."
                "[/cyan]"
            )
            deleted_count = await memory_manager.cleanup_old_memories(days)

            if deleted_count > 0:
                console.print(
                    f"[green]Successfully deleted {deleted_count} old "
                    "memories[/green]"
                )
            else:
                console.print(
                    "[yellow]No old memories found to delete[/yellow]"
                )

        except Exception as e:
            console.print(f"[red]Cleanup failed:[/red] {e}")

        finally:
            await memory_manager.close()

    asyncio.run(cleanup_memories())


@app.command()
def performance(
    show_live: bool = typer.Option(
        False, "--live", "-l", help="Show live performance metrics"
    ),
    component: Optional[str] = typer.Option(
        None, "--component", "-c", help="Filter by component"
    ),
    minutes: int = typer.Option(
        10, "--minutes", "-m", help="Minutes of history to show"
    ),
    export: Optional[str] = typer.Option(
        None, "--export", "-e", help="Export metrics to file"
    ),
):
    """Display performance metrics and system status."""
    import asyncio
    from .performance.monitor import PerformanceMonitor, ComponentType
    from rich.table import Table
    from rich.live import Live
    import time
    
    async def show_performance():
        monitor = PerformanceMonitor()
        
        if show_live:
            console.print("[cyan]Starting live performance monitoring (Press Ctrl+C to stop)...[/cyan]")
            
            def generate_live_display():
                summary = monitor.get_performance_summary()
                
                # Create main table
                table = Table(title="Live Performance Metrics")
                table.add_column("Component", style="cyan")
                table.add_column("Avg Latency (ms)", style="green")
                table.add_column("Memory (MB)", style="yellow")
                table.add_column("CPU %", style="red")
                table.add_column("Count", style="blue")
                
                for comp_name, stats in summary.get("components", {}).items():
                    table.add_row(
                        comp_name,
                        f"{stats.get('avg_latency', 0):.2f}",
                        f"{stats.get('avg_memory', 0):.1f}",
                        f"{stats.get('avg_cpu', 0):.1f}",
                        str(stats.get('count', 0))
                    )
                
                return table
            
            try:
                with Live(generate_live_display(), refresh_per_second=2) as live:
                    while True:
                        await asyncio.sleep(0.5)
                        live.update(generate_live_display())
            except KeyboardInterrupt:
                console.print("\n[yellow]Live monitoring stopped[/yellow]")
        
        else:
            # Show static performance summary
            console.print("[cyan]Performance Summary[/cyan]")
            
            summary = monitor.get_performance_summary()
            
            # Overall stats
            console.print(f"Total metrics collected: {summary.get('total_metrics', 0)}")
            console.print(f"Monitoring active: {summary.get('monitoring_active', False)}")
            
            # Component stats table
            if summary.get("components"):
                table = Table(title="Component Performance")
                table.add_column("Component", style="cyan")
                table.add_column("Count", style="blue")
                table.add_column("Avg Latency (ms)", style="green")
                table.add_column("Min/Max (ms)", style="yellow")
                table.add_column("Memory (MB)", style="red")
                
                for comp_name, stats in summary["components"].items():
                    if component and comp_name != component:
                        continue
                    
                    table.add_row(
                        comp_name,
                        str(stats.get('count', 0)),
                        f"{stats.get('avg_latency', 0):.2f}",
                        f"{stats.get('min_latency', 0):.1f}/{stats.get('max_latency', 0):.1f}",
                        f"{stats.get('avg_memory', 0):.1f}"
                    )
                
                console.print(table)
            else:
                console.print("[yellow]No performance metrics available[/yellow]")
            
            # Current resources
            if "current_resources" in summary:
                resources = summary["current_resources"]
                console.print(f"\n[cyan]Current Resources:[/cyan]")
                console.print(f"Memory: {resources.get('memory_percent', 0):.1f}% "
                           f"({resources.get('memory_used_mb', 0):.0f}MB used)")
                console.print(f"CPU: {resources.get('cpu_percent', 0):.1f}%")
        
        # Export if requested
        if export:
            comp_filter = ComponentType(component) if component else None
            await monitor.export_metrics(export, component=comp_filter)
            console.print(f"[green]Performance metrics exported to {export}[/green]")
    
    asyncio.run(show_performance())


@app.callback()
def callback():
    """
    Transcriber - AI Voice Agent
    
    Natural voice interface for AI tool execution with session management.
    """
    pass


async def run_voice_agent(model: str):
    """Run the voice agent with full pipeline."""
    from .config import settings
    from .pipeline import TextModeAgent, run_voice_pipeline
    
    # Update model setting
    settings.agent.model = model
    
    console.print("[yellow]Initializing voice pipeline...[/yellow]")
    
    try:
        # Check if we can run full voice pipeline or fallback to text mode
        try:
            await run_voice_pipeline(settings)
        except Exception as voice_error:
            console.print(f"[red]Voice pipeline failed: {voice_error}[/red]")
            console.print("[yellow]Falling back to text-only mode...[/yellow]")
            
            # Try text-only mode
            text_agent = TextModeAgent(settings)
            await text_agent.initialize()
            await text_agent.run_text_mode()
        
    except Exception as e:
        console.print(f"[red]❌ Fatal error: {e}[/red]")
        import traceback
        if settings.debug:
            console.print(f"[red]{traceback.format_exc()}[/red]")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()