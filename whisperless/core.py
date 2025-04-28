import json
from pathlib import Path
from typing import List, Tuple, Dict
import whisper
from rich.console import Console
from rich.progress import Progress
from .ebook import parse_epub
from .audio import AudioProcessor

console = Console()

def parse_chunk_size(chunk_size: str) -> int:
    """Convert chunk size string (e.g. '5m') to seconds"""
    if chunk_size.endswith('m'):
        return int(chunk_size[:-1]) * 60
    elif chunk_size.endswith('s'):
        return int(chunk_size[:-1])
    else:
        return int(chunk_size)

def transcribe_audio(audio_path: str, chunk_size: str = "5m") -> Dict:
    """Transcribe audio using whisper base model.
    
    For MP3 files (chapters), processes the whole file at once.
    For M4B files, processes in chunks of chunk_size.
    """
    console.print("[bold]Loading whisper base model...[/bold]")
    model = whisper.load_model("base")
    
    path = Path(audio_path)
    
    # Process MP3 files as whole chapters (no chunking)
    if path.suffix.lower() == '.mp3':
        console.print("[bold]Processing MP3 chapter (whole file)...[/bold]")
        # Get audio features for silence detection
        processor = AudioProcessor(path)
        features = processor.process_chapter()
        
        # Transcribe whole file
        try:
            result = model.transcribe(
                str(path),
                verbose=True,  # show progress
                word_timestamps=True  # get word-level timestamps
            )
            
            # Convert to our format with silence region info
            words = []
            for segment in result["segments"]:
                # Find nearest silence regions for validation
                segment_start = segment["start"]
                segment_end = segment["end"]
                
                # Add word-level info
                for word in segment["words"]:
                    words.append({
                        "word": word["word"].strip(),
                        "start": word["start"],
                        "end": word["end"],
                        "confidence": word.get("confidence", 1.0)
                    })
            
            return {
                "words": words,
                "silent_regions": features.silent_regions,
                "duration": features.duration,
                "text": result["text"]
            }
        except Exception as e:
            console.print(f"[red]Error during transcription: {e}[/red]")
            raise
    
    # Process M4B files in chunks (5m default)
    elif path.suffix.lower() == '.m4b':
        chunk_seconds = parse_chunk_size(chunk_size)
        console.print(f"[bold]Processing M4B in {chunk_seconds}s chunks...[/bold]")
        
        processor = AudioProcessor(path)
        features = processor.process_chapter()
        
        # Split audio into chunks
        samples, sample_rate = processor.get_numpy_array()
        chunk_samples = chunk_seconds * sample_rate
        
        words = []
        total_duration = 0
        
        for i in range(0, len(samples), chunk_samples):
            chunk = samples[i:i + chunk_samples]
            chunk_duration = len(chunk) / sample_rate
            
            console.print(f"[bold]Processing chunk {i//chunk_samples + 1}...[/bold]")
            
            try:
                # Save chunk to temporary file
                temp_path = path.parent / f"temp_chunk_{i//chunk_samples}.mp3"
                processor.export_audio(temp_path)
                
                # Transcribe chunk
                result = model.transcribe(
                    str(temp_path),
                    verbose=True,
                    word_timestamps=True
                )
                
                # Adjust timestamps for chunk position
                for segment in result["segments"]:
                    for word in segment["words"]:
                        words.append({
                            "word": word["word"].strip(),
                            "start": word["start"] + total_duration,
                            "end": word["end"] + total_duration,
                            "confidence": word.get("confidence", 1.0)
                        })
                
                total_duration += chunk_duration
                
                # Clean up temp file
                temp_path.unlink()
                
            except Exception as e:
                console.print(f"[red]Error processing chunk {i//chunk_samples + 1}: {e}[/red]")
                raise
        
        return {
            "words": words,
            "silent_regions": features.silent_regions,
            "duration": total_duration,
            "text": " ".join(word["word"] for word in words)
        }
    
    else:
        raise ValueError(f"Unsupported audio format: {path.suffix}")

def split_ebook(ebook_path: str) -> List[str]:
    """Split ebook into sentences"""
    return parse_epub(ebook_path)

def match_text(audio_words: List[Dict], ebook_sentences: List[str]) -> Dict[str, Tuple[float, float]]:
    """Match audio words with ebook sentences"""
    # TODO: implement fuzzy matching
    return {}

def save_alignment(alignment: Dict[str, Tuple[float, float]], output_path: str):
    """Save alignment to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(alignment, f, indent=2) 