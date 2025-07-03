"""
Simple voice pipeline using available components (GTTS + Mock STT).
"""

import asyncio
import logging
from rich.console import Console
from rich.panel import Panel

from .audio.capture import AudioCapture
from .audio.vad import VADProcessor
from .audio.gtts_tts import GTTSService
from .audio.mock_stt import MockSTTProcessor
from .agent.text_agent import TextOnlyAgent
from .config import Settings

logger = logging.getLogger(__name__)
console = Console()


class SimpleVoicePipeline:
    """Simple voice pipeline using mock STT and GTTS TTS."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.audio_capture = AudioCapture(settings.audio)
        self.vad_processor = VADProcessor(settings.voice.vad_threshold)
        self.stt_processor = MockSTTProcessor(settings)  # Using mock STT
        self.tts_service = GTTSService(settings)  # Using GTTS
        self.agent = TextOnlyAgent(settings)
        
        self.running = False
        
    async def initialize(self) -> None:
        """Initialize all components."""
        console.print("[yellow]Initializing simple voice pipeline...[/yellow]")
        
        try:
            # Initialize TTS (GTTS)
            await self.tts_service.initialize()
            console.print("[green]✅ TTS (GTTS) initialized[/green]")
            
            # Initialize Mock STT
            await self.stt_processor.initialize()
            console.print("[green]✅ Mock STT initialized[/green]")
            
            # Initialize Agent
            await self.agent.initialize()
            console.print("[green]✅ Agent initialized[/green]")
            
            console.print("[green]✅ Simple voice pipeline ready![/green]")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    async def run_demo(self) -> None:
        """Run a demo of the voice pipeline."""
        console.print(Panel.fit(
            "[bold green]Simple Voice Pipeline Demo[/bold green]\n"
            "Using: Mock STT + GTTS TTS + AI Agent",
            border_style="green"
        ))
        
        try:
            # Simulate some voice interactions
            demo_inputs = [
                "Hello, how are you?",
                "What's the weather like?",
                "Tell me a joke",
                "Thank you!"
            ]
            
            for i, mock_input in enumerate(demo_inputs, 1):
                console.print(f"\n[blue]🎤 Simulated Voice Input {i}:[/blue] {mock_input}")
                
                # Process through agent
                console.print("[green]🤖 Agent Response:[/green] ", end="")
                
                response_text = ""
                async for chunk in self.agent.process_text_input_stream(mock_input):
                    console.print(chunk, end="", style="green")
                    response_text += chunk
                
                console.print()
                
                # Generate TTS (mock audio output)
                console.print("[cyan]🔊 Generating speech...[/cyan]")
                audio_data = await self.tts_service.speak(response_text)
                console.print(f"[dim]Generated {len(audio_data)} audio samples[/dim]")
                
                # Small delay between interactions
                await asyncio.sleep(1)
            
            console.print("\n[green]✅ Voice pipeline demo completed![/green]")
            
        except Exception as e:
            console.print(f"[red]Demo error: {e}[/red]")
            raise
    
    async def run_audio_test(self) -> None:
        """Test audio capture with mock STT."""
        console.print("[yellow]Testing audio capture with mock STT...[/yellow]")
        
        try:
            async with self.audio_capture:
                console.print("[green]🎤 Audio capture started[/green]")
                console.print("[dim]Speak into your microphone for 10 seconds...[/dim]")
                
                # Get audio stream
                audio_stream = self.audio_capture.get_audio_chunks()
                
                # Process through VAD
                speech_segments = self.vad_processor.process_audio_stream(audio_stream)
                
                # Process through mock STT
                segment_count = 0
                async for text in self.stt_processor.process_audio_segments(speech_segments):
                    segment_count += 1
                    console.print(f"[blue]🎤 Mock STT {segment_count}:[/blue] {text}")
                    
                    # Process through agent
                    console.print("[green]🤖 Agent:[/green] ", end="")
                    response_text = ""
                    async for chunk in self.agent.process_text_input_stream(text):
                        console.print(chunk, end="", style="green")
                        response_text += chunk
                    console.print()
                    
                    # Generate TTS
                    audio_data = await self.tts_service.speak(response_text)
                    console.print(f"[cyan]🔊 Generated {len(audio_data)} audio samples[/cyan]")
                    
                    # Stop after first interaction for demo
                    break
                
                console.print("[green]✅ Audio test completed![/green]")
                
        except Exception as e:
            console.print(f"[red]Audio test error: {e}[/red]")
            raise
    
    async def cleanup(self) -> None:
        """Clean up all components."""
        try:
            await self.tts_service.cleanup()
            await self.stt_processor.cleanup()
            await self.agent.cleanup()
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


async def run_simple_voice_demo(settings: Settings) -> None:
    """Run the simple voice pipeline demo."""
    pipeline = SimpleVoicePipeline(settings)
    
    try:
        await pipeline.initialize()
        
        # Run the demo
        await pipeline.run_demo()
        
        # Optionally test audio (commented out to avoid hanging)
        # await pipeline.run_audio_test()
        
    except Exception as e:
        console.print(f"[red]Pipeline error: {e}[/red]")
        import traceback
        traceback.print_exc()
    finally:
        await pipeline.cleanup()


if __name__ == "__main__":
    from .config import settings
    asyncio.run(run_simple_voice_demo(settings))