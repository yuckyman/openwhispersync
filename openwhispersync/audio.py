from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from pydub import AudioSegment
import numpy as np
import librosa
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AudioFeatures:
    """Container for extracted audio features."""
    # basic properties
    path: Path  # path to the audio file
    duration: float  # in seconds
    sample_rate: int
    
    # amplitude features
    rms_energy: np.ndarray  # root mean square energy
    
    # spectral features
    mfcc: np.ndarray  # mel-frequency cepstral coefficients
    spectral_centroid: np.ndarray
    zero_crossing_rate: np.ndarray
    
    # silence detection
    silent_regions: List[tuple]  # list of (start, end) in seconds

class AudioProcessor:
    """Handles audio file processing for both MP3 and M4B formats."""
    
    def __init__(self, audio_path: Union[str, Path]):
        self.audio_path = Path(audio_path)
        self.audio: Optional[AudioSegment] = None
        self._load_audio()
    
    def _load_audio(self) -> None:
        """Load the audio file, supporting both MP3 and M4B formats."""
        try:
            if self.audio_path.suffix.lower() == '.m4b':
                # TODO: implement M4B support with chunking
                raise NotImplementedError("M4B support coming soon!")
            
            self.audio = AudioSegment.from_mp3(self.audio_path)
            logger.info(f"Loaded audio file: {self.audio_path}")
            logger.info(f"Duration: {self.audio.duration_seconds:.2f}s")
            logger.info(f"Sample rate: {self.audio.frame_rate}Hz")
            
        except Exception as e:
            logger.error(f"Failed to load audio file: {e}")
            raise
    
    def get_numpy_array(self) -> tuple[np.ndarray, int]:
        """Convert audio to numpy array for processing."""
        samples = np.array(self.audio.get_array_of_samples())
        
        # convert to float32 and normalize
        if self.audio.sample_width == 2:  # 16-bit audio
            samples = samples.astype(np.float32) / 32768.0
        elif self.audio.sample_width == 4:  # 32-bit audio
            samples = samples.astype(np.float32) / 2147483648.0
        
        return samples, self.audio.frame_rate
    
    def process_chapter(self) -> AudioFeatures:
        """Process a single chapter file and extract audio features."""
        if not self.audio:
            raise ValueError("No audio loaded")
        
        # convert to numpy array
        samples, sample_rate = self.get_numpy_array()
        
        # calculate features
        logger.info("Extracting audio features...")
        
        # amplitude features
        rms_energy = librosa.feature.rms(y=samples)[0]
        
        # spectral features
        mfcc = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=samples, sr=sample_rate)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(samples)[0]
        
        # silence detection (threshold at -50dB)
        silent_regions = librosa.effects.split(
            samples, 
            top_db=50,
            frame_length=2048,
            hop_length=512
        )
        # convert frame indices to seconds
        silent_regions = [(start/sample_rate, end/sample_rate) 
                         for start, end in silent_regions]
        
        logger.info("Feature extraction complete!")
        
        return AudioFeatures(
            path=self.audio_path,
            duration=self.audio.duration_seconds,
            sample_rate=sample_rate,
            rms_energy=rms_energy,
            mfcc=mfcc,
            spectral_centroid=spectral_centroid,
            zero_crossing_rate=zero_crossing_rate,
            silent_regions=silent_regions
        )
    
    def export_audio(self, output_path: Union[str, Path]) -> None:
        """Export the processed audio to the specified path."""
        try:
            if not self.audio:
                raise ValueError("No audio loaded")
            
            self.audio.export(output_path, format="mp3")
            logger.info(f"Exported audio to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to export audio: {e}")
            raise 