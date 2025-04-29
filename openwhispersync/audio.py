from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from pydub import AudioSegment
import numpy as np
import librosa
import logging
from dataclasses import dataclass
import whisper
import glob
import json

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
    
    # transcription
    transcription: Optional[Dict] = None

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
        non_silent_regions = librosa.effects.split(
            samples, 
            top_db=40,
            frame_length=2048,
            hop_length=512
        )
        
        # convert non-silent regions to silent regions by finding gaps
        silent_regions = []
        if len(non_silent_regions) > 0:
            # add initial silence if it exists
            if non_silent_regions[0][0] > 0:
                silent_regions.append((0, non_silent_regions[0][0]/sample_rate))
            
            # add silence between non-silent regions
            for i in range(len(non_silent_regions)-1):
                silent_regions.append((
                    non_silent_regions[i][1]/sample_rate,
                    non_silent_regions[i+1][0]/sample_rate
                ))
            
            # add final silence if it exists
            if non_silent_regions[-1][1] < len(samples):
                silent_regions.append((
                    non_silent_regions[-1][1]/sample_rate,
                    len(samples)/sample_rate
                ))
        
        logger.info(f"Found {len(silent_regions)} silent regions")
        for i, (start, end) in enumerate(silent_regions):
            logger.info(f"  Silent region {i+1}: {start:.2f}s - {end:.2f}s")
        
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

class AudioTranscriber:
    """Handles transcription of multiple audio files using whisper."""
    
    def __init__(self, model_name: str = "base"):
        self.model = whisper.load_model(model_name)
        self.processors: Dict[str, AudioProcessor] = {}
        
    def load_directory(self, directory: Union[str, Path]) -> None:
        """Load all mp3 files from a directory."""
        directory = Path(directory)
        mp3_files = sorted(glob.glob(str(directory / "*.mp3")))
        
        logger.info(f"Found {len(mp3_files)} mp3 files in {directory}")
        
        for mp3_file in mp3_files:
            processor = AudioProcessor(mp3_file)
            self.processors[mp3_file] = processor
            
    def transcribe_all(self) -> Dict[str, Dict]:
        """Transcribe all loaded audio files."""
        transcriptions = {}
        
        for path, processor in self.processors.items():
            logger.info(f"Transcribing {path}...")
            
            # get audio as numpy array
            samples, sample_rate = processor.get_numpy_array()
            
            # transcribe
            result = self.model.transcribe(samples)
            
            # store result
            transcriptions[path] = result
            
            # add to processor's features
            features = processor.process_chapter()
            features.transcription = result
            
        return transcriptions
    
    def save_transcriptions(self, output_file: Union[str, Path]) -> None:
        """Save all transcriptions to a json file."""
        transcriptions = self.transcribe_all()
        
        with open(output_file, 'w') as f:
            json.dump(transcriptions, f, indent=2)
            
        logger.info(f"Saved transcriptions to {output_file}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python -m openwhispersync.audio <audio_directory> <output_file>")
        sys.exit(1)
        
    audio_dir = sys.argv[1]
    output_file = sys.argv[2]
    
    transcriber = AudioTranscriber()
    transcriber.load_directory(audio_dir)
    transcriber.save_transcriptions(output_file) 