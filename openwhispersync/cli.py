import click
import json
from rich.console import Console
from rich.progress import Progress
from .core import (
    transcribe_audio,
    split_ebook,
    match_text,
    save_alignment,
    process_all_chapters,
    match_chapters
)
from .ebook import parse_epub
from rich.table import Table

console = Console()

@click.group()
def main():
    """OpenWhisperSync: Lightweight audiobook alignment"""
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
    
    sentences, chapter_markers = parse_epub(ebook)
    
    # save sentences and chapter markers to json
    with open(out, 'w') as f:
        json.dump({
            "sentences": sentences,
            "chapter_markers": chapter_markers
        }, f, indent=2)
    
    console.print(f"[green]✓[/green] Extracted {len(sentences)} sentences and {len(chapter_markers)} chapter markers to [bold]{out}[/bold]")
    
    # print chapter markers
    table = Table(title="Chapter Markers")
    table.add_column("Sentence Index", justify="right", style="cyan")
    table.add_column("Marker", style="magenta")
    
    for idx, marker in chapter_markers.items():
        table.add_row(str(idx), marker)
    console.print(table)

@main.command()
@click.option("--audio-dir", required=True, help="Directory containing MP3 chapter files")
@click.option("--out", default="transcriptions.json", help="Output JSON file for transcriptions")
def transcribe(audio_dir, out):
    """Transcribe all audio chapters"""
    chapters = process_all_chapters(audio_dir, out)
    console.print(f"[green]✓[/green] Transcribed {len(chapters)} chapters! Output saved to [bold]{out}[/bold]")

@main.command()
@click.option("--transcriptions", required=True, help="Path to transcriptions JSON file")
@click.option("--ebook", required=True, help="Path to ebook file")
@click.option("--out-dir", default="alignments", help="Directory to save alignment results")
def align(transcriptions, ebook, out_dir):
    """Align transcribed audio with ebook text"""
    with Progress() as progress:
        task = progress.add_task("[cyan]Matching chapters...", total=None)
        match_chapters(transcriptions, ebook, out_dir)
        progress.update(task, completed=True)
    
    console.print(f"[green]✓[/green] Alignment complete! Results saved to [bold]{out_dir}[/bold]")

@main.command()
@click.option("--audio", required=True, help="Path to audio file")
@click.option("--alignment", required=True, help="Path to alignment JSON file")
@click.option("--out-dir", default="visualizations", help="Directory to save plots")
@click.option("--show", is_flag=True, help="Show plots interactively")
def visualize(audio, alignment, out_dir, show):
    """Visualize audio alignment results"""
    from pathlib import Path
    from .visualize import plot_alignment, plot_silence_regions, plot_alignment_confidence
    
    # create output directory
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # generate plots
    plot_alignment(
        audio,
        alignment,
        output_path=out_dir / "alignment.png",
        show_plot=show
    )
    
    # load alignment data for confidence plot
    with open(alignment) as f:
        alignments = json.load(f)
    
    plot_alignment_confidence(
        alignments,
        output_path=out_dir / "confidence.png",
        show_plot=show
    )
    
    console.print(f"[green]✓[/green] Generated visualizations in [bold]{out_dir}[/bold]")

@main.command()
@click.option("--alignment", required=True, help="Path to alignment JSON file")
@click.option("--audio", required=True, help="Path to audio file")
@click.option("--output", default="karaoke.html", help="Path to save HTML file")
def karaoke(alignment, audio, output):
    """Create a karaoke-style viewer for aligned text"""
    from .karaoke import create_karaoke_viewer
    create_karaoke_viewer(alignment, audio, output)
    console.print(f"[green]✓[/green] Karaoke viewer created at [bold]{output}[/bold]")

if __name__ == "__main__":
    main() 