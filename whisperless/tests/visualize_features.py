import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from whisperless.audio import AudioProcessor

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_features(features, output_path: Path):
    """Create a multi-panel visualization of audio features."""
    # create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    fig.suptitle('Audio Feature Analysis', fontsize=16)
    
    # time axis for all plots
    time = np.linspace(0, features.duration, len(features.rms_energy))
    
    # 1. waveform with silent regions
    processor = AudioProcessor(features.path)
    samples, _ = processor.get_numpy_array()
    # downsample for visualization
    samples = samples[::100]  # take every 100th sample
    time_wave = np.linspace(0, features.duration, len(samples))
    axes[0].plot(time_wave, samples)
    for start, end in features.silent_regions:
        axes[0].axvspan(start, end, color='red', alpha=0.3)
    axes[0].set_title('Waveform with Silent Regions')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    
    # 2. MFCCs
    mfcc_plot = axes[1].imshow(features.mfcc, aspect='auto', origin='lower')
    axes[1].set_title('MFCCs')
    axes[1].set_xlabel('Time (frames)')
    axes[1].set_ylabel('MFCC Coefficients')
    plt.colorbar(mfcc_plot, ax=axes[1])
    
    # 3. spectral features
    axes[2].plot(time, features.spectral_centroid, label='Spectral Centroid')
    axes[2].plot(time, features.zero_crossing_rate, label='Zero Crossing Rate')
    axes[2].set_title('Spectral Features')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Value')
    axes[2].legend()
    
    # 4. RMS energy
    axes[3].plot(time, features.rms_energy)
    axes[3].set_title('RMS Energy')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Energy')
    
    # adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Saved visualization to: {output_path}")

def main():
    # paths
    chapter_path = Path("whisperless/files/frankenstein/frankenstein_09_shelley_64kb.mp3")
    output_path = Path("whisperless/tests/output/features.png")
    
    # process chapter
    processor = AudioProcessor(chapter_path)
    features = processor.process_chapter()
    
    # create visualization
    plot_features(features, output_path)

if __name__ == "__main__":
    main() 