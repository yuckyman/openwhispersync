import sys
from pathlib import Path

# Add the parent directory to the path so we can import whisperless
sys.path.append(str(Path(__file__).parent.parent))

from whisperless.core import transcribe_audio
from rich.console import Console

console = Console()

def main():
    # Test with chapter 1 of frankenstein
    audio_path = Path(__file__).parent.parent / "files" / "frankenstein" / "frankenstein_02_shelley_64kb.mp3"
    
    console.print("[bold green]Starting whisper transcription test...[/bold green]")
    console.print(f"Processing: {audio_path}")
    
    try:
        result = transcribe_audio(str(audio_path))
        
        # Print some basic stats
        console.print("\n[bold]Transcription Results:[/bold]")
        console.print(f"Total words: {len(result['words'])}")
        console.print(f"Duration: {result['duration']:.2f} seconds")
        console.print(f"Silent regions detected: {len(result['silent_regions'])}")
        
        # Print first few words with timestamps
        console.print("\n[bold]First few words:[/bold]")
        for word in result['words'][:10]:
            console.print(f"{word['start']:.2f}s - {word['word']} (confidence: {word['confidence']:.2f})")
        
        # Print first few silent regions
        console.print("\n[bold]First few silent regions:[/bold]")
        for start, end in result['silent_regions'][:5]:
            console.print(f"{start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
        
    except Exception as e:
        console.print(f"[red]Error during test: {e}[/red]")
        raise

if __name__ == "__main__":
    main() 