"""
Voice pipeline that orchestrates the complete STT → Agent → TTS flow.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from .agent.core import AgentState, VoiceAgent
from .audio.capture import AudioCapture
from .audio.output import AudioOutput
from .audio.tts import TTSService
from .audio.vad import VADProcessor
from .config import Settings

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class PipelineStats:
    """Statistics for the voice pipeline."""
    audio_chunks_processed: int = 0
    speech_segments_detected: int = 0
    transcriptions_completed: int = 0
    responses_generated: int = 0
    total_latency_ms: float = 0.0
    last_interaction_time: Optional[datetime] = None


class VoicePipeline:
    """Complete voice pipeline orchestrator."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
        # Initialize components
        self.audio_capture = AudioCapture(settings.audio)
        self.vad_processor = VADProcessor(settings.voice.vad_threshold)
        self.audio_output = AudioOutput(settings.audio)
        self.tts_service = TTSService(settings)
        self.voice_agent = VoiceAgent(settings)
        
        # Pipeline state
        self.running = False
        self.stats = PipelineStats()
        self._tasks: set = set()
        
        # UI components
        self.console = Console()
        self._live_display: Optional[Live] = None
        
    async def initialize(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Initializing voice pipeline...")
        
        try:
            # Initialize TTS service
            await self.tts_service.initialize()
            
            # Initialize voice agent
            await self.voice_agent.initialize()
            
            # Setup agent state callback
            self.voice_agent.add_state_callback(self._on_agent_state_change)
            
            logger.info("Voice pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    async def start(self) -> None:
        """Start the voice pipeline."""
        if self.running:
            logger.warning("Pipeline already running")
            return
        
        logger.info("Starting voice pipeline...")
        self.running = True
        
        try:
            # Start display
            self._start_display()
            
            # Start audio capture
            await self.audio_capture.__aenter__()
            
            # Start the main pipeline loop
            await self._run_pipeline()
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise
        finally:
            await self.stop()
    
    async def stop(self) -> None:
        """Stop the voice pipeline."""
        if not self.running:
            return
        
        logger.info("Stopping voice pipeline...")
        self.running = False
        
        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Cleanup components
        try:
            await self.audio_capture.__aexit__(None, None, None)
            await self.tts_service.cleanup()
            await self.voice_agent.cleanup()
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
        
        # Stop display
        self._stop_display()
        
        logger.info("Voice pipeline stopped")
    
    async def _run_pipeline(self) -> None:
        """Main pipeline loop."""
        try:
            # Get audio stream
            audio_stream = self.audio_capture.get_audio_chunks()
            
            # Process through VAD
            speech_segments = self.vad_processor.process_audio_stream(audio_stream)
            
            # Process through voice agent and TTS
            async for response_text in self.voice_agent.process_audio_segments(speech_segments):
                if response_text.strip():
                    # Generate TTS audio
                    tts_audio = await self.tts_service.speak(response_text)
                    
                    # Play audio response
                    if len(tts_audio) > 0:
                        await self.audio_output.play_audio(tts_audio)
                    
                    self.stats.responses_generated += 1
                    self.stats.last_interaction_time = datetime.now()
                    
                    # Update display
                    self._update_display()
                    
                    # Short pause between responses
                    await asyncio.sleep(0.1)
        
        except asyncio.CancelledError:
            logger.info("Pipeline loop cancelled")
        except Exception as e:
            logger.error(f"Pipeline loop error: {e}")
            raise
    
    def _on_agent_state_change(self, old_state: AgentState, new_state: AgentState) -> None:
        """Handle agent state changes."""
        logger.debug(f"Agent state: {old_state} -> {new_state}")
        
        # Update stats based on state
        if new_state == AgentState.PROCESSING:
            self.stats.transcriptions_completed += 1
        
        self._update_display()
    
    def _start_display(self) -> None:
        """Start the live display."""
        if self._live_display is None:
            self._live_display = Live(
                self._generate_display(),
                console=self.console,
                refresh_per_second=4,
                vertical_overflow="visible"
            )
            self._live_display.start()
    
    def _stop_display(self) -> None:
        """Stop the live display."""
        if self._live_display:
            self._live_display.stop()
            self._live_display = None
    
    def _update_display(self) -> None:
        """Update the live display."""
        if self._live_display:
            self._live_display.update(self._generate_display())
    
    def _generate_display(self) -> Panel:
        """Generate the display content."""
        # Status indicators
        status_text = Text()
        
        # Agent state
        state_color = {
            AgentState.IDLE: "green",
            AgentState.LISTENING: "blue", 
            AgentState.PROCESSING: "yellow",
            AgentState.RESPONDING: "magenta",
            AgentState.ERROR: "red"
        }.get(self.voice_agent.state, "white")
        
        status_text.append("Agent: ", style="bold")
        status_text.append(self.voice_agent.state.value.title(), style=f"bold {state_color}")
        status_text.append(" | ")
        
        # Audio level
        if hasattr(self.audio_capture, 'get_audio_level'):
            level = self.audio_capture.get_audio_level()
            bar_length = int(level * 10)
            bar = "█" * bar_length + "░" * (10 - bar_length)
            status_text.append("Audio: ", style="bold")
            status_text.append(f"|{bar}| {level:.2f}", style="cyan")
        
        # Recent conversation
        conversation_text = Text()
        recent_messages = self.voice_agent.get_conversation_history()[-4:]  # Last 4 messages
        
        for msg in recent_messages:
            if msg.message_type == "user":
                conversation_text.append(f"🎤 You: {msg.content}\n", style="bold blue")
            elif msg.message_type == "assistant":
                conversation_text.append(f"🤖 Agent: {msg.content}\n", style="bold green")
        
        if not conversation_text.plain:
            conversation_text.append("No conversation yet. Start speaking!", style="dim")
        
        # Stats
        stats_text = Text()
        stats_text.append(f"Audio chunks: {self.stats.audio_chunks_processed} | ", style="dim")
        stats_text.append(f"Speech segments: {self.stats.speech_segments_detected} | ", style="dim")
        stats_text.append(f"Transcriptions: {self.stats.transcriptions_completed} | ", style="dim")
        stats_text.append(f"Responses: {self.stats.responses_generated}", style="dim")
        
        if self.stats.last_interaction_time:
            elapsed = (datetime.now() - self.stats.last_interaction_time).total_seconds()
            stats_text.append(f" | Last: {elapsed:.1f}s ago", style="dim")
        
        # Combine all sections
        content = Columns([
            Text("🎙️  Voice Interface Active", style="bold green"),
            status_text
        ])
        
        display_content = Text()
        display_content.append(content)
        display_content.append("\n\n")
        display_content.append("Recent Conversation:", style="bold")
        display_content.append("\n")
        display_content.append(conversation_text)
        display_content.append("\n")
        display_content.append(stats_text)
        display_content.append("\n\n")
        display_content.append("Press Ctrl+C to exit", style="dim italic")
        
        return Panel(
            display_content,
            title="[bold green]Transcriber AI Voice Agent[/bold green]",
            border_style="green",
            padding=(1, 2)
        )
    
    async def health_check(self) -> dict[str, Any]:
        """Check the health of all pipeline components."""
        health = {
            "pipeline": {"status": "running" if self.running else "stopped"},
            "agent": await self.voice_agent.health_check(),
            "tts": {"status": "initialized" if self.tts_service.processor._initialized else "not_initialized"},
            "audio": {
                "capture": "active" if hasattr(self.audio_capture, '_stream') else "inactive",
                "output": "ready"
            },
            "stats": {
                "audio_chunks": self.stats.audio_chunks_processed,
                "speech_segments": self.stats.speech_segments_detected,
                "transcriptions": self.stats.transcriptions_completed,
                "responses": self.stats.responses_generated
            }
        }
        
        return health


async def run_voice_pipeline(settings: Settings) -> None:
    """Run the complete voice pipeline."""
    pipeline = VoicePipeline(settings)
    
    try:
        # Initialize pipeline
        await pipeline.initialize()
        
        # Start pipeline
        await pipeline.start()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise
    finally:
        await pipeline.stop()


class TextModeAgent:
    """Fallback text-only mode for testing."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.voice_agent = VoiceAgent(settings)
        
    async def initialize(self) -> None:
        """Initialize text mode agent."""
        await self.voice_agent.initialize()
        
    async def run_text_mode(self) -> None:
        """Run in text-only mode."""
        console.print("[yellow]Running in text-only mode for testing[/yellow]")
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
                async for chunk in self.voice_agent.process_text_input_stream(user_input):
                    console.print(chunk, end="", style="green")
                
                console.print()  # New line after response
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
        
        console.print("[yellow]Goodbye![/yellow]")
        await self.voice_agent.cleanup()