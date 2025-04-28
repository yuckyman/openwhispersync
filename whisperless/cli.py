import click
import json
from rich.console import Console
from rich.progress import Progress
from .core import (
    transcribe_audio,
    split_ebook,
    match_text,
    save_alignment
)
from .ebook import parse_epub

console = Console()

@click.group()
def main():
    """Whisperless: Lightweight audiobook alignment"""
    pass

@main.command()
@click.option("--audio", required=True, help="Path to audio file")
@click.option("--ebook", required=True, help="Path to ebook file")
@click.option("--out", default="sync.json", help="Output JSON file")
@click.option("--chunk-size", default="5m", help="Processing chunk size (e.g. 5m for 5 minutes)")
def align(audio, ebook, out, chunk_size):
    """Align audio with ebook text"""
    with Progress() as progress:
        # Step 1: Transcribe
        task1 = progress.add_task("[cyan]Transcribing audio...", total=None)
        audio_words = transcribe_audio(audio, chunk_size)
        progress.update(task1, completed=True)
        
        # Step 2: Split ebook
        task2 = progress.add_task("[cyan]Processing ebook...", total=None)
        ebook_sentences = split_ebook(ebook)
        progress.update(task2, completed=True)
        
        # Step 3: Match
        task3 = progress.add_task("[cyan]Matching text...", total=None)
        alignment = match_text(audio_words, ebook_sentences)
        progress.update(task3, completed=True)
        
        # Step 4: Save
        task4 = progress.add_task("[cyan]Saving alignment...", total=None)
        save_alignment(alignment, out)
        progress.update(task4, completed=True)
    
    console.print(f"[green]✓[/green] Alignment complete! Output saved to [bold]{out}[/bold]")

@main.command()
@click.option("--ebook", required=True, help="Path to ebook file")
@click.option("--out", default="sentences.json", help="Output JSON file")
def parse(ebook, out):
    """Parse ebook and extract sentences (for testing)"""
    console.print(f"[bold]Parsing {ebook}[/bold]")
    
    sentences = parse_epub(ebook)
    
    # save sentences to json
    with open(out, 'w') as f:
        json.dump(sentences, f, indent=2)
    
    console.print(f"[green]✓[/green] Extracted {len(sentences)} sentences to [bold]{out}[/bold]")

if __name__ == "__main__":
    main() 