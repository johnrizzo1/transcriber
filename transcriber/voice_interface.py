"""
Real-time voice interface for the transcriber.
"""

import asyncio
import logging
import numpy as np
import sys
import select
import termios
import tty
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.text import Text

from .audio.capture import AudioCapture
from .audio.vad import VADProcessor
from .audio.playback import AudioPlayback
from .audio.edge_tts import EdgeTTSService
from .audio.whisper_stt import WhisperSTTService
from .audio.echo_cancellation import AudioLevelFilter
from .agent.core import VoiceAgent
from .config import Settings

logger = logging.getLogger(__name__)
console = Console()


class KeyboardMonitor:
    """Monitor for keyboard input to interrupt speech."""
    
    def __init__(self):
        self.stop_requested = False
        self.old_settings = None
        
    def __enter__(self):
        """Set up non-blocking keyboard input."""
        if sys.stdin.isatty():
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore terminal settings."""
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def check_for_stop(self) -> bool:
        """Check if 'q' was pressed to stop speaking."""
        if not sys.stdin.isatty():
            return False
            
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            char = sys.stdin.read(1)
            if char.lower() == 'q':
                self.stop_requested = True
                return True
        return False
    
    def reset(self):
        """Reset the stop flag."""
        self.stop_requested = False


class VoiceInterface:
    """Real-time voice interface with audio capture and playback."""
    
    def __init__(self, settings: Settings, device_id: Optional[int] = None):
        self.settings = settings
        self.device_id = device_id
        
        # Update settings with device_id if provided
        if device_id is not None:
            settings.audio.input_device = device_id
            settings.audio.output_device = device_id
        
        # Audio components
        self.audio_capture = AudioCapture(settings.audio)
        self.audio_playback = AudioPlayback(settings.audio, device_id=device_id)
        self.vad_processor = VADProcessor(settings.voice.vad_threshold)
        
        # Speech components
        self.stt_processor = WhisperSTTService(settings)
        self.tts_service = EdgeTTSService(settings)
        
        # Audio filtering
        self.audio_filter = AudioLevelFilter()
        
        # Agent
        self.agent = VoiceAgent(settings)
        
        # State
        self.running = False
        self.processing = False
        self.speaking = False
        
        # Keyboard monitor for interrupting speech
        self.keyboard_monitor = KeyboardMonitor()
        
    async def initialize(self) -> None:
        """Initialize all components."""
        console.print("[yellow]🎤 Initializing voice interface...[/yellow]")
        
        try:
            # Initialize TTS
            await self.tts_service.initialize()
            console.print("[green]✅ TTS initialized[/green]")
            
            # Initialize STT
            await self.stt_processor.initialize()
            console.print("[green]✅ STT initialized[/green]")
            
            # Initialize Agent
            await self.agent.initialize()
            console.print("[green]✅ Agent initialized[/green]")
            
            # Initialize audio playback
            await self.audio_playback.initialize()
            console.print("[green]✅ Audio playback initialized[/green]")
            
            console.print("[green]✅ Voice interface ready![/green]")
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise
    
    def create_status_layout(self) -> Layout:
        """Create the status display layout."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", size=10),
            Layout(name="status", size=3)
        )
        
        # Header
        layout["header"].update(Panel(
            "[bold cyan]🎤 Transcriber Voice Interface[/bold cyan]\n"
            "[dim]Speak naturally - the AI is listening...[/dim]",
            border_style="cyan"
        ))
        
        # Main area for conversation
        layout["main"].update(Panel(
            "[dim]Conversation will appear here...[/dim]",
            title="Conversation",
            border_style="blue"
        ))
        
        # Status
        layout["status"].update(Panel(
            "🟢 Listening | Press Ctrl+C to exit",
            border_style="green"
        ))
        
        return layout
    
    async def process_audio_pipeline(self) -> None:
        """Process the audio pipeline with real-time speech recognition."""
        try:
            chunk_count = 0
            audio_buffer = []
            silence_count = 0
            
            async for chunk in self.audio_capture.get_audio_chunks():
                if not self.running:
                    break
                
                chunk_count += 1
                
                # Calculate audio level
                audio_level = np.sqrt(np.mean(chunk**2))
                
                # Skip processing while speaking (but don't mute - too aggressive)
                if self.speaking:
                    if audio_buffer:
                        audio_buffer = []
                        silence_count = 0
                    if chunk_count % 30 == 0:
                        console.print(f"\r🔊 Speaking...", end="")
                    continue
                
                # Simple, reliable speech detection with lower threshold
                speech_threshold = 0.02  # More sensitive threshold
                is_speech = audio_level > speech_threshold
                
                if is_speech:
                    audio_buffer.append(chunk)
                    silence_count = 0
                    
                    # Show speech detection immediately
                    if chunk_count % 10 == 0:
                        level_bars = int(min(audio_level / speech_threshold, 1.0) * 15)
                        bars = "█" * level_bars + "░" * (15 - level_bars)
                        console.print(f"\r🎤 [{bars}] {audio_level:.3f}", end="")
                else:
                    # Count silence chunks
                    if len(audio_buffer) > 0:
                        silence_count += 1
                    
                    # Show listening status less frequently
                    if chunk_count % 50 == 0:
                        console.print(f"\r🎧 Listening...", end="")
                    
                    # Process speech after SHORT silence (faster response)
                    if len(audio_buffer) > 5 and silence_count > 8:  # ~0.8 seconds of silence
                        await self._process_speech_segment(audio_buffer)
                        audio_buffer = []
                        silence_count = 0
                        console.print()
                
                # Prevent very long recordings
                if len(audio_buffer) > 80:  # ~8 seconds max
                    await self._process_speech_segment(audio_buffer)
                    audio_buffer = []
                    silence_count = 0
                    console.print()
                    
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            console.print(f"\n[red]Error: {e}[/red]")
    
    async def _process_speech_segment(self, audio_buffer: list) -> None:
        """Process a segment of speech audio."""
        if not audio_buffer:
            return
        
        # Prevent concurrent processing
        if self.processing or self.speaking:
            return
        
        try:
            self.processing = True
            
            # Combine audio chunks
            audio_data = np.concatenate(audio_buffer)
            
            console.print(f"\n[cyan]🔄 Processing {len(audio_data)/16000:.1f}s...[/cyan]")
            
            # Transcribe with Whisper
            text = await self.stt_processor.transcribe(audio_data)
            
            if text and text.strip():
                console.print(f"[blue]You:[/blue] {text}")
                
                # Process through agent
                console.print("[green]AI:[/green] ", end="")
                response_text = ""
                
                async for chunk in self.agent.process_text_input_stream(text):
                    console.print(chunk, end="", style="green")
                    response_text += chunk
                
                console.print()
                
                # Speak response
                if response_text:
                    await self._speak_response(response_text)
                
                console.print()
            else:
                console.print("[dim]No speech detected[/dim]")
                
        except Exception as e:
            logger.error(f"Speech processing error: {e}")
            console.print(f"[red]Error: {e}[/red]")
        finally:
            self.processing = False
    
    async def _speak_response(self, response_text: str) -> None:
        """Handle TTS generation and playback with interrupt capability."""
        try:
            console.print("[dim]Speaking... (press 'q' to interrupt)[/dim]")
            
            # Mute audio input FIRST to prevent feedback
            self.audio_capture.mute()
            self.speaking = True
            self.keyboard_monitor.reset()
            
            # Generate TTS audio
            audio_data = await self.tts_service.speak(response_text)
            
            if len(audio_data) > 0:
                # Start playback task
                playback_task = asyncio.create_task(self.audio_playback.play(audio_data))
                
                # Monitor for keyboard interrupt while playing
                try:
                    while not playback_task.done():
                        if self.keyboard_monitor.check_for_stop():
                            console.print("\n[yellow]⏹️  Speech interrupted by user[/yellow]")
                            # Interrupt the playback cleanly
                            self.audio_playback.interrupt()
                            break
                        await asyncio.sleep(0.1)  # Check every 100ms
                    
                    # Wait for playback to complete (either normally or after interrupt)
                    await playback_task
                    
                except Exception as e:
                    # If there's an error, make sure we interrupt the playback
                    self.audio_playback.interrupt()
                    logger.error(f"Playback monitoring error: {e}")
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            console.print(f"[red]TTS error: {e}[/red]")
        finally:
            # Always unmute and clear speaking flag
            self.speaking = False
            self.audio_capture.unmute()
    
    async def run(self) -> None:
        """Run the voice interface."""
        self.running = True
        
        console.print(Panel.fit(
            "[bold green]🎤 Voice Interface Active[/bold green]\n"
            "Speak naturally - the AI is listening!\n"
            "[dim]Press 'q' during speech to interrupt | Press Ctrl+C to exit[/dim]",
            border_style="green"
        ))
        
        try:
            with self.keyboard_monitor:
                async with self.audio_capture:
                    console.print("\n[cyan]🎧 Listening...[/cyan]\n")
                    
                    # Run the audio pipeline
                    await self.process_audio_pipeline()
                    
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping voice interface...[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise
        finally:
            self.running = False
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            await self.audio_playback.cleanup()
            await self.tts_service.cleanup()
            await self.stt_processor.cleanup()
            await self.agent.cleanup()
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


class SimpleVoiceLoop:
    """Simplified voice loop for testing."""
    
    def __init__(self, settings: Settings, device_id: Optional[int] = None):
        self.settings = settings
        self.device_id = device_id
        self.interface = VoiceInterface(settings, device_id)
        
    async def run(self) -> None:
        """Run a simple voice loop."""
        try:
            # Initialize
            await self.interface.initialize()
            
            # Show instructions
            console.print("\n" + "="*50)
            console.print("[bold cyan]🎤 Real-Time Voice Chat[/bold cyan]")
            console.print("="*50)
            console.print("\n[yellow]Instructions:[/yellow]")
            console.print("• Speak clearly into your microphone")
            console.print("• The AI will respond with voice")
            console.print("• Press 'q' while AI is speaking to interrupt")
            console.print("• Say 'goodbye' or press Ctrl+C to exit")
            console.print("\n[dim]Note: Using Whisper STT + Edge TTS (neural voices)[/dim]\n")
            
            # Run the interface
            await self.interface.run()
            
        except Exception as e:
            console.print(f"[red]Voice interface error: {e}[/red]")
            import traceback
            traceback.print_exc()
        finally:
            await self.interface.cleanup()
            console.print("\n[green]✅ Voice interface closed[/green]")


async def run_voice_interface(settings: Settings, device_id: Optional[int] = None) -> None:
    """Run the voice interface."""
    voice_loop = SimpleVoiceLoop(settings, device_id)
    await voice_loop.run()


if __name__ == "__main__":
    from .config import settings
    asyncio.run(run_voice_interface(settings))