import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple
import librosa
import librosa.display
from rich.console import Console

console = Console()

# create visualizations directory if it doesn't exist
VISUALIZATIONS_DIR = Path("openwhispersync/visualizations")
VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

def plot_alignment(audio_path: str, 
                  alignment_path: str,
                  output_path: str = str(VISUALIZATIONS_DIR / "alignment.png"),
                  show_plot: bool = False):
    """
    Create a visualization of audio alignment with ebook text.
    
    Args:
        audio_path: Path to audio file
        alignment_path: Path to alignment JSON file
        output_path: Path to save plot (defaults to visualizations/alignment.png)
        show_plot: Whether to display the plot
    """
    # load audio
    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # load alignment data
    with open(alignment_path) as f:
        alignments = json.load(f)
    
    # create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[1, 2])
    
    # plot waveform
    librosa.display.waveshow(y, sr=sr, ax=ax1)
    ax1.set_title('Audio Waveform with Aligned Sentences')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    
    # plot alignment bars
    for i, align in enumerate(alignments):
        # create color based on confidence
        confidence = align['confidence']
        color = plt.cm.viridis(confidence)  # viridis colormap
        
        # plot alignment bar
        ax1.axvspan(
            align['start_time'],
            align['end_time'],
            alpha=0.3,
            color=color
        )
        
        # add sentence text (truncated for readability)
        text = align['sentence'][:50] + '...' if len(align['sentence']) > 50 else align['sentence']
        ax1.text(
            align['start_time'],
            0.8,
            text,
            rotation=90,
            va='top',
            ha='right',
            fontsize=8
        )
    
    # plot confidence scores
    times = [align['start_time'] for align in alignments]
    confidences = [align['confidence'] for align in alignments]
    ax2.plot(times, confidences, 'o-')
    ax2.set_title('Alignment Confidence Scores')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Confidence')
    ax2.set_ylim(0, 1)
    
    # save and show
    plt.tight_layout()
    plt.savefig(output_path)
    console.print(f"✓ Saved alignment plot to {output_path}")
    
    if show_plot:
        plt.show()
    plt.close()

def plot_silence_regions(audio_path: str,
                        silent_regions: List[Tuple[float, float]],
                        output_path: str = str(VISUALIZATIONS_DIR / "silence.png"),
                        show_plot: bool = False):
    """
    Plot audio waveform with silent regions highlighted.
    
    Args:
        audio_path: Path to audio file
        silent_regions: List of (start, end) silence timestamps
        output_path: Path to save plot (defaults to visualizations/silence.png)
        show_plot: Whether to display the plot
    """
    # load audio
    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # create figure
    plt.figure(figsize=(15, 5))
    
    # plot waveform
    librosa.display.waveshow(y, sr=sr)
    
    # highlight silent regions
    for start, end in silent_regions:
        plt.axvspan(start, end, color='red', alpha=0.3)
    
    plt.title('Audio Waveform with Silent Regions')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.xlim(0, duration)
    plt.grid(True, alpha=0.3)
    
    # save or show plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    console.print(f"[green]✓[/green] Saved silence plot to [bold]{output_path}[/bold]")
    
    if show_plot:
        plt.show()
    
    plt.close()

def plot_alignment_confidence(alignments: List[Dict],
                           output_path: str = str(VISUALIZATIONS_DIR / "confidence.png"),
                           show_plot: bool = False):
    """
    Plot confidence scores for each alignment.
    
    Args:
        alignments: List of alignment dictionaries
        output_path: Path to save plot (defaults to visualizations/confidence.png)
        show_plot: Whether to display the plot
    """
    # extract confidence scores and timing info
    confidences = [a['confidence'] for a in alignments]
    start_times = [a['start'] for a in alignments]
    durations = [a['end'] - a['start'] for a in alignments]
    
    # create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), height_ratios=[1, 1])
    
    # plot 1: confidence scores
    ax1.plot(confidences, 'o-', alpha=0.7, label='Confidence Score')
    ax1.axhline(y=0.6, color='r', linestyle='--', alpha=0.5, label='Min Confidence')
    ax1.set_title('Alignment Confidence Scores')
    ax1.set_xlabel('Sentence Index')
    ax1.set_ylabel('Confidence Score')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.legend()
    
    # plot 2: temporal alignment
    for i, (start, duration) in enumerate(zip(start_times, durations)):
        color = plt.cm.viridis(confidences[i])
        ax2.barh(i, duration, left=start, color=color, alpha=0.6)
    
    ax2.set_title('Temporal Alignment')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Sentence Index')
    ax2.grid(True, alpha=0.3)
    
    # add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis')
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2)
    cbar.set_label('Confidence Score')
    
    # adjust layout
    plt.tight_layout()
    
    # save or show plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    console.print(f"[green]✓[/green] Saved confidence plot to [bold]{output_path}[/bold]")
    
    if show_plot:
        plt.show()
    
    plt.close() 