import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple
import librosa
import librosa.display
from rich.console import Console
from pydub import AudioSegment
import textwrap

# IEEE-friendly matplotlib style settings
plt.style.use('seaborn-v0_8-paper')  # Clean, academic style
plt.rcParams.update({
    'font.family': 'serif',  # Use serif font for better readability
    'font.size': 10,         # Base font size
    'axes.labelsize': 10,    # Axis label size
    'axes.titlesize': 12,    # Title size
    'xtick.labelsize': 9,    # X-axis tick label size
    'ytick.labelsize': 9,    # Y-axis tick label size
    'legend.fontsize': 9,    # Legend font size
    'figure.figsize': (6, 4), # IEEE-friendly figure size
    'figure.dpi': 300,       # High resolution
    'savefig.dpi': 300,      # High resolution for saved figures
    'savefig.format': 'pdf', # Save as PDF for better quality
    'savefig.bbox': 'tight', # Tight bounding box
    'axes.grid': True,       # Show grid
    'grid.alpha': 0.3,       # Semi-transparent grid
    'lines.linewidth': 1.5,  # Slightly thicker lines
    'axes.linewidth': 0.8,   # Slightly thicker axes
    'xtick.major.width': 0.8,# Slightly thicker ticks
    'ytick.major.width': 0.8,# Slightly thicker ticks
    'xtick.minor.width': 0.6,# Slightly thicker minor ticks
    'ytick.minor.width': 0.6,# Slightly thicker minor ticks
    'xtick.major.size': 4,   # Slightly longer ticks
    'ytick.major.size': 4,   # Slightly longer ticks
    'xtick.minor.size': 2,   # Slightly longer minor ticks
    'ytick.minor.size': 2,   # Slightly longer minor ticks
    'axes.spines.right': False,  # Remove right spine
    'axes.spines.top': False,    # Remove top spine
})

console = Console()

# create visualizations directory if it doesn't exist
VISUALIZATIONS_DIR = Path("openwhispersync/visualizations")
VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

def _load_audio_segment(audio_path, zoom_start=None, zoom_duration=4.0):
    """Helper to load full audio or a zoomed segment."""
    sr = None
    y = None
    actual_duration = 0
    time_offset = 0

    if zoom_start is not None:
        try:
            y, sr = librosa.load(audio_path, sr=None, offset=zoom_start, duration=zoom_duration)
            actual_duration = librosa.get_duration(y=y, sr=sr)
            time_offset = zoom_start
            print(f"Loaded audio segment: {time_offset:.2f}s - {time_offset + actual_duration:.2f}s")
            if actual_duration < zoom_duration:
                 print(f"  Note: Actual duration ({actual_duration:.2f}s) is less than requested ({zoom_duration:.2f}s), possibly due to reaching end of file.")
        except Exception as e:
            print(f"Error loading zoomed audio segment: {e}. Loading full audio.")
            zoom_start = None # Fallback

    if zoom_start is None: # Either initially None or fallback
        y, sr = librosa.load(audio_path, sr=None)
        actual_duration = librosa.get_duration(y=y, sr=sr)
        time_offset = 0
        print("Loaded full audio.")

    return y, sr, actual_duration, time_offset, zoom_start is not None

def plot_alignment(audio_path, alignments, output_path=None, show=False, zoom_start=None, zoom_duration=4.0):
    """Plot audio waveform with aligned sentences, potentially zoomed."""
    # Load alignment data if it's a file path
    if isinstance(alignments, str):
        try:
            with open(alignments) as f:
                alignments = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
             print(f"Error loading alignments: {e}")
             return

    if not alignments:
        print("No alignment data for plot_alignment.")
        return

    # Load audio (full or segment)
    y, sr, actual_duration, time_offset, is_zoomed = _load_audio_segment(audio_path, zoom_start, zoom_duration)
    zoom_end = time_offset + actual_duration

    # Filter and adjust alignments for the view window
    visible_alignments = []
    for a in alignments:
        # Check for overlap with the time window (time_offset to zoom_end)
        if max(a['start_time'], time_offset) < min(a['end_time'], zoom_end):
            adj_a = a.copy()
            # Clip and adjust times relative to the window start (time_offset)
            adj_a['start_time'] = max(0, a['start_time'] - time_offset)
            adj_a['end_time'] = min(actual_duration, a['end_time'] - time_offset)
            visible_alignments.append(adj_a)

    if not visible_alignments and is_zoomed:
        print(f"No alignments fall within the zoom window {time_offset:.2f}s - {zoom_end:.2f}s.")
        # Optionally, you might still want to plot the empty waveform
        # return # Uncomment to skip plotting if no alignments in zoom

    # Create plot with IEEE-friendly settings
    fig, ax = plt.subplots(figsize=(6, 4))  # IEEE-friendly size

    # Plot waveform with IEEE-friendly colors
    times = np.arange(len(y)) / sr
    ax.plot(times, y, color='#1f77b4', alpha=0.7)  # IEEE-friendly blue

    # Plot sentence segments with confidence-based colors
    cmap = plt.cm.viridis

    for i, alignment in enumerate(visible_alignments):
        start = alignment['start_time']
        end = alignment['end_time']
        conf = alignment['confidence']
        sentence = alignment['sentence']

        # Color based on confidence
        color = cmap(conf)

        # Plot colored segment
        ax.axvspan(start, end, color=color, alpha=0.4)

        # Add truncated sentence text with IEEE-friendly formatting
        wrapped = textwrap.shorten(sentence, width=50, placeholder="...")
        y_pos = 0.9 - (i % 5) * 0.1
        ax.text(start, y_pos, wrapped, fontsize=8,
                horizontalalignment='left', verticalalignment='top',
                transform=ax.get_yaxis_transform())

    # Add colorbar with IEEE-friendly formatting
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Confidence', fontsize=9)

    # Labels and formatting with IEEE-friendly style
    if is_zoomed:
        ax.set_xlabel(f'Time (relative to {time_offset:.2f}s)', fontsize=10)
        ax.set_title(f'Audio Alignment ({time_offset:.2f}s - {zoom_end:.2f}s)', fontsize=12)
    else:
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_title('Audio Alignment with Text', fontsize=12)

    ax.set_ylabel('Amplitude', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)

    # Save or show with IEEE-friendly settings
    if output_path:
        # Change extension to .pdf for better quality
        pdf_path = str(output_path).replace('.png', '.pdf')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        print(f"Alignment plot saved to {pdf_path}")

    if show:
        plt.show()

    plt.close(fig)

def plot_silence_regions(audio_path, output_path=None, show=False, zoom_start=None, zoom_duration=4.0):
    """
    Plot audio waveform with silent regions highlighted.
    Can zoom into a specific time window.
    (Includes previous zoom implementation - slightly refactored for consistency)
    """
    # Load audio (full or segment)
    y, sr, actual_duration, time_offset, is_zoomed = _load_audio_segment(audio_path, zoom_start, zoom_duration)
    zoom_end = time_offset + actual_duration

    # Detect silent regions (on the original full audio)
    try:
        from .audio import AudioProcessor # Import locally if needed
        processor = AudioProcessor(audio_path)
        features = processor.process_chapter()
        all_silent_regions = features.silent_regions
    except Exception as e:
        print(f"Could not process audio for silence detection: {e}")
        all_silent_regions = []

    # Filter and adjust silent regions for the zoom window
    visible_silent_regions = []
    if all_silent_regions: # Check if list is not empty
        for start, end in all_silent_regions:
            if max(start, time_offset) < min(end, zoom_end):
                visible_start = max(0, start - time_offset)
                visible_end = min(actual_duration, end - time_offset)
                visible_silent_regions.append((visible_start, visible_end))

    # Create plot with IEEE-friendly settings
    fig, ax = plt.subplots(figsize=(6, 4))  # IEEE-friendly size

    # Plot waveform with IEEE-friendly colors
    times = np.arange(len(y)) / sr
    ax.plot(times, y, color='#1f77b4', alpha=0.7)  # IEEE-friendly blue

    # Highlight silent regions with IEEE-friendly colors
    for start, end in visible_silent_regions:
        ax.axvspan(start, end, color='#d62728', alpha=0.3)  # IEEE-friendly red

    # Labels and formatting with IEEE-friendly style
    if is_zoomed:
        plot_title = f'Audio Waveform with Silent Regions ({time_offset:.2f}s - {zoom_end:.2f}s)'
        xlabel = f'Time (relative to {time_offset:.2f}s)'
    else:
        plot_title = f'Audio Waveform with Silent Regions ({len(visible_silent_regions)} regions)'
        xlabel = 'Time (s)'

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Amplitude', fontsize=10)
    ax.set_title(plot_title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(min(y.min(), -0.1), max(y.max(), 0.1))

    # Save or show with IEEE-friendly settings
    if output_path:
        # Change extension to .pdf for better quality
        pdf_path = str(output_path).replace('.png', '.pdf')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        print(f"Silence plot saved to {pdf_path}")

    if show:
        plt.show()

    plt.close(fig)

def plot_alignment_confidence(audio_path, alignments, output_path=None, show=False, zoom_start=None, zoom_duration=4.0):
    """Plot alignment confidence scores over time, potentially zoomed."""
     # Load alignment data if it's a file path
    if isinstance(alignments, str):
        try:
            with open(alignments) as f:
                alignments = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
             print(f"Error loading alignments: {e}")
             return

    if not alignments:
        print("No alignment data for plot_alignment_confidence.")
        return

    # Determine time window
    is_zoomed = zoom_start is not None
    time_offset = zoom_start if is_zoomed else 0
    # Estimate full duration if not zooming, or use zoom duration
    # We need an end time for filtering even if not plotting waveform
    full_duration_est = max(a['end_time'] for a in alignments) if alignments else 0
    actual_duration = zoom_duration if is_zoomed else full_duration_est
    # If actual_duration from audio load is needed, load audio first
    # For now, use zoom_duration or estimate

    zoom_end = time_offset + actual_duration


    # Filter and adjust alignments for the view window
    visible_alignments = []
    for a in alignments:
        if max(a['start_time'], time_offset) < min(a['end_time'], zoom_end):
            adj_a = a.copy()
            adj_a['start_time'] = max(0, a['start_time'] - time_offset)
            adj_a['end_time'] = min(actual_duration, a['end_time'] - time_offset)
            # Ensure start is not after end after clipping
            if adj_a['start_time'] < adj_a['end_time']:
                 visible_alignments.append(adj_a)


    if not visible_alignments and is_zoomed:
        print(f"No alignments fall within the confidence plot zoom window {time_offset:.2f}s - {zoom_end:.2f}s.")
        # Optionally return or plot empty graph
        # return


    # Create plot with IEEE-friendly settings
    fig, ax = plt.subplots(figsize=(6, 4))  # IEEE-friendly size

    # Create time axis and confidence data from VISIBLE alignments
    times = []
    conf_values = []
    if visible_alignments:
        # Sort by start time just in case
        visible_alignments.sort(key=lambda x: x['start_time'])
        for alignment in visible_alignments:
            start = alignment['start_time'] # Already adjusted
            end = alignment['end_time']     # Already adjusted
            conf = alignment['confidence']
            # Add points at start and end of each alignment segment within the window
            times.extend([start, end])
            conf_values.extend([conf, conf])

        # Plot confidence line with IEEE-friendly colors
        ax.plot(times, conf_values, '-', color='#1f77b4', linewidth=1.5, label='Alignment Confidence')
    else:
         # Plot empty if no visible alignments
         ax.plot([], [], '-', color='#1f77b4', linewidth=1.5, label='Alignment Confidence')


    # Add min confidence threshold line with IEEE-friendly colors
    # Make this configurable maybe?
    min_conf_threshold = 0.7
    ax.axhline(y=min_conf_threshold, color='#d62728', linestyle='--', 
               label=f'Min Confidence ({min_conf_threshold})')

    # Add audio waveform background with IEEE-friendly colors
    if audio_path:
        try:
            y_wav, sr_wav, dur_wav, offset_wav, _ = _load_audio_segment(audio_path, zoom_start, zoom_duration)
            if y_wav is not None and len(y_wav) > 0:
                 audio_times = np.linspace(0, dur_wav, len(y_wav)) # Relative times
                 # Normalize waveform for plotting in background
                 samples = y_wav / max(abs(y_wav.max()), abs(y_wav.min()), 1e-6) * 0.3 # Scale and prevent div by zero
                 ax.plot(audio_times, samples, '-', color='#2ca02c', alpha=0.1, label='_nolegend_') # Hide from legend
        except Exception as e:
            print(f"Could not add waveform to confidence plot: {e}")

    # Labels and formatting with IEEE-friendly style
    if is_zoomed:
        ax.set_xlabel(f'Time (relative to {time_offset:.2f}s)', fontsize=10)
        ax.set_title(f'Alignment Confidence ({time_offset:.2f}s - {zoom_end:.2f}s)', fontsize=12)
    else:
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_title('Alignment Confidence Over Time', fontsize=12)

    ax.set_ylabel('Confidence', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)

    # Save or show with IEEE-friendly settings
    if output_path:
        # Change extension to .pdf for better quality
        pdf_path = str(output_path).replace('.png', '.pdf')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        print(f"Confidence plot saved to {pdf_path}")

    if show:
        plt.show()

    plt.close(fig)

def plot_alignment_scatter(alignments, output_path=None, show=False, title="Alignment Scatter Plot"):
    """
    Plots aligned sentences as points (audio time vs ebook sentence index).

    Args:
        alignments (list): List of alignment dictionaries with 'start_time' and 'sentence_idx'.
        output_path (str, optional): Path to save the plot image. Defaults to None.
        show (bool, optional): Whether to display the plot interactively. Defaults to False.
        title (str, optional): Title for the plot.
    """
    if isinstance(alignments, str):
        try:
            with open(alignments) as f:
                alignments = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
             print(f"Error loading alignments: {e}")
             return

    if not alignments:
        print("No alignment data to plot.")
        return

    # Extract data points
    # Ensure sentence_idx exists and is numeric
    start_times = []
    sentence_indices = []
    for a in alignments:
        if 'start_time' in a and 'sentence_idx' in a and isinstance(a['sentence_idx'], (int, float)):
             start_times.append(a['start_time'])
             sentence_indices.append(a['sentence_idx'])

    if not start_times:
        print("No valid data points found in alignments for scatter plot (missing 'start_time' or numeric 'sentence_idx'?).")
        return

    # Create plot with IEEE-friendly settings
    fig, ax = plt.subplots(figsize=(6, 4))  # IEEE-friendly size

    # Plot scatter with IEEE-friendly colors and markers
    scatter = ax.scatter(start_times, sentence_indices, 
                        s=20,  # Slightly larger markers
                        alpha=0.7, 
                        color='#1f77b4',  # IEEE-friendly blue
                        edgecolors='none',  # No edge color for cleaner look
                        marker='o')  # Circle markers

    # Add a trend line if there's enough data
    if len(start_times) > 1:
        z = np.polyfit(start_times, sentence_indices, 1)
        p = np.poly1d(z)
        ax.plot(start_times, p(start_times), 
                color='#d62728',  # IEEE-friendly red
                linestyle='--',
                linewidth=1.5,
                alpha=0.7,
                label='Linear Trend')

    # Labels and formatting with IEEE-friendly style
    ax.set_xlabel("Audio Time (seconds)", fontsize=10)
    ax.set_ylabel("Ebook Sentence Index", fontsize=10)
    ax.set_title(title, fontsize=12)
    
    # Add grid with IEEE-friendly style
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Format x-axis to show minutes:seconds
    from matplotlib.ticker import FuncFormatter
    def format_time(x, pos):
        mins = int(x // 60)
        secs = int(x % 60)
        return f'{mins:02d}:{secs:02d}'
    ax.xaxis.set_major_formatter(FuncFormatter(format_time))
    
    # Add legend if we have a trend line
    if len(start_times) > 1:
        ax.legend(fontsize=9, loc='upper left')

    # Save or show with IEEE-friendly settings
    if output_path:
        # Change extension to .pdf for better quality
        pdf_path = str(output_path).replace('.png', '.pdf')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to {pdf_path}")

    if show:
        plt.show()

    plt.close(fig)  # Close the figure to free memory 