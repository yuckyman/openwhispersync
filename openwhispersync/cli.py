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
import os.path
from pathlib import Path
import matplotlib.pyplot as plt

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
    # Check if files exist
    if not os.path.isfile(transcriptions):
        console.print(f"[red]Error:[/red] Transcriptions file not found: [bold]{transcriptions}[/bold]")
        console.print(f"[yellow]Tip:[/yellow] Make sure the path is correct and the file exists.")
        return
        
    if not os.path.isfile(ebook):
        console.print(f"[red]Error:[/red] Ebook file not found: [bold]{ebook}[/bold]")
        console.print(f"[yellow]Tip:[/yellow] Make sure the path is correct and the file exists.")
        return
    
    # Create output directory if it doesn't exist
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    match_chapters(transcriptions, ebook, out_dir)

@main.command()
@click.option("--audio", required=True, help="Path to audio file")
@click.option("--alignment", required=True, help="Path to alignment JSON file")
@click.option("--out-dir", default="visualizations", help="Directory to save plots")
@click.option("--show", is_flag=True, help="Show plots interactively")
@click.option("--zoom-start", type=float, default=None, help="Start time (seconds) for time-based plot zoom")
@click.option("--zoom-duration", type=float, default=4.0, help="Duration (seconds) for time-based plot zoom")
def visualize(audio, alignment, out_dir, show, zoom_start, zoom_duration):
    """Visualize audio alignment results"""
    from pathlib import Path
    from .visualize import plot_alignment, plot_silence_regions, plot_alignment_confidence, plot_alignment_scatter
    
    # create output directory
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Load alignment data once ---
    alignment_data = None
    try:
        with open(alignment) as f:
            alignment_data = json.load(f)
        if not alignment_data:
             console.print(f"[yellow]Warning:[/yellow] Alignment file '{alignment}' is empty.")
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] Alignment file not found: [bold]{alignment}[/bold]")
        return # Exit if alignment file is missing
    except json.JSONDecodeError:
         console.print(f"[red]Error:[/red] Could not decode alignment JSON: [bold]{alignment}[/bold]")
         return # Exit if JSON is invalid

    # --- Plot Alignment ---
    console.print("Generating Alignment Plot...")
    plot_alignment(
        audio_path=audio,
        alignments=alignment_data, # Pass loaded data
        output_path=out_dir / "alignment.png",
        show=show,
        zoom_start=zoom_start, # Pass zoom params
        zoom_duration=zoom_duration # Pass zoom params
    )

    # --- Plot Silence Regions ---
    console.print("Generating Silence Regions Plot...")
    plot_silence_regions(
        audio_path=audio,
        output_path=out_dir / "silence.png",
        show=show,
        zoom_start=zoom_start,
        zoom_duration=zoom_duration
    )

    # --- Plot Alignment Confidence ---
    console.print("Generating Alignment Confidence Plot...")
    plot_alignment_confidence(
        audio_path=audio,
        alignments=alignment_data, # Pass loaded data
        output_path=out_dir / "confidence.png",
        show=show,
        zoom_start=zoom_start, # Pass zoom params
        zoom_duration=zoom_duration # Pass zoom params
    )

    # --- Plot Alignment Scatter ---
    # No zoom for scatter plot
    console.print("Generating Alignment Scatter Plot...")
    plot_alignment_scatter(
        alignments=alignment_data, # Pass loaded data
        output_path=out_dir / "alignment_scatter.png",
        show=show,
        title=f"Alignment Scatter: {Path(audio).name}"
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