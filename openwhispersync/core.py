import json
from pathlib import Path
from typing import List, Tuple, Dict
import whisper
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from .ebook import parse_epub
from .audio import AudioProcessor
from .matcher import TextMatcher

console = Console()

def parse_chunk_size(chunk_size: str) -> int:
    """Convert chunk size string (e.g. '5m') to seconds"""
    if chunk_size.endswith('m'):
        return int(chunk_size[:-1]) * 60
    elif chunk_size.endswith('s'):
        return int(chunk_size[:-1])
    else:
        return int(chunk_size)

def transcribe_audio(audio_path: str, chunk_size: str = "5m") -> List[Dict]:
    """Transcribe audio file using whisper."""
    # TODO: implement whisper transcription
    return []

def split_ebook(ebook_path: str) -> List[str]:
    """Split ebook into sentences."""
    return parse_epub(ebook_path)

def match_text(audio_words: List[Dict], 
              ebook_sentences: List[str],
              audio_path: str = None) -> List[Dict]:
    """
    Match transcribed audio with ebook text.
    
    Args:
        audio_words: List of word dicts with 'text', 'start', 'end'
        ebook_sentences: List of sentences from ebook
        audio_path: Optional path to audio file for silence detection
        
    Returns:
        List of alignment dicts with 'sentence', 'start', 'end', 'confidence'
    """
    # get silent regions if audio path provided
    silent_regions = None
    if audio_path:
        processor = AudioProcessor(audio_path)
        features = processor.process_chapter()
        silent_regions = features.silent_regions
    
    # create matcher and get results
    matcher = TextMatcher()
    results = matcher.match(audio_words, ebook_sentences, silent_regions)
    
    # convert to dict format
    return [
        {
            'sentence': r.sentence,
            'start': r.start_time,
            'end': r.end_time,
            'confidence': r.confidence,
            'matched_text': r.matched_text,
            'is_silence_based': r.is_silence_based
        }
        for r in results
    ]

def save_alignment(alignment: List[Dict], output_path: str):
    """Save alignment data to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(alignment, f, indent=2)

def process_all_chapters(audio_dir: str, output_path: str):
    """
    Process all MP3 chapters in a directory and save transcriptions with chapter info.
    
    Args:
        audio_dir: Directory containing MP3 chapter files
        output_path: Path to save JSON output
    """
    import json
    from pathlib import Path
    import whisper
    from rich.progress import Progress, SpinnerColumn, TextColumn
    
    # get all mp3 files in directory
    audio_dir = Path(audio_dir)
    mp3_files = sorted(audio_dir.glob("*.mp3"))
    
    if not mp3_files:
        raise ValueError(f"No MP3 files found in {audio_dir}")
    
    # load whisper model
    model = whisper.load_model("base", device="cpu", in_memory=True)
    model = model.float()  # convert to fp32
    
    # process each chapter
    chapters = []
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    )
    
    with progress:
        task = progress.add_task("[cyan]Processing chapters...", total=len(mp3_files))
        
        for mp3_path in mp3_files:
            # extract chapter number from filename (e.g., "frankenstein_01_shelley_64kb.mp3")
            chapter_num = int(mp3_path.stem.split("_")[1])
            
            # transcribe audio
            result = model.transcribe(
                str(mp3_path),
                word_timestamps=True,
                language="en"
            )
            
            # convert whisper segments to our format
            words = []
            for segment in result["segments"]:
                for word in segment["words"]:
                    words.append({
                        "text": word["word"].strip(),
                        "start": word["start"],
                        "end": word["end"]
                    })
            
            # calculate duration from last word's end time
            duration = words[-1]["end"] if words else 0
            
            # add chapter info
            chapters.append({
                "number": chapter_num,
                "filename": mp3_path.name,
                "duration": duration,
                "word_count": len(words),
                "words": words
            })
            
            progress.update(task, advance=1)
    
    # save results
    with open(output_path, 'w') as f:
        json.dump({
            "book": audio_dir.name,  # use directory name as book title
            "total_chapters": len(chapters),
            "total_words": sum(c["word_count"] for c in chapters),
            "chapters": chapters
        }, f, indent=2)
    
    return chapters

def match_chapters(audio_json: str, ebook_path: str, output_dir: str):
    """
    Match transcribed audio chapters with ebook text.
    
    Args:
        audio_json: Path to JSON file with audio transcriptions
        ebook_path: Path to ebook file
        output_dir: Directory to save alignment results
    """
    import json
    from pathlib import Path
    from .ebook import parse_epub
    from .matcher import TextMatcher
    
    # load audio transcriptions
    with open(audio_json) as f:
        audio_data = json.load(f)
    
    # parse ebook
    ebook_sentences = parse_epub(ebook_path)
    
    # create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # match each chapter
    for chapter in audio_data["chapters"]:
        # find chapter start in ebook
        chapter_start = 0
        for i, sentence in enumerate(ebook_sentences):
            if f"chapter {chapter['number']}" in sentence.lower():
                chapter_start = i + 1
                break
        
        # get chapter sentences
        chapter_sentences = ebook_sentences[chapter_start:]
        
        # create matcher
        matcher = TextMatcher()
        
        # match text
        results = matcher.match(chapter["words"], chapter_sentences)
        
        # save results
        output_path = output_dir / f"chapter_{chapter['number']}_alignment.json"
        with open(output_path, 'w') as f:
            json.dump([
                {
                    'sentence': r.sentence,
                    'start': r.start_time,
                    'end': r.end_time,
                    'confidence': r.confidence,
                    'matched_text': r.matched_text,
                    'is_silence_based': r.is_silence_based
                }
                for r in results
            ], f, indent=2) 