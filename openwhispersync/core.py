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
    # use process_all_chapters to handle the transcription
    chapters = process_all_chapters(audio_path, None)
    if not chapters:
        return []
    return chapters[0]["words"]

def split_ebook(ebook_path: str) -> Tuple[List[str], Dict[int, str]]:
    """Split ebook into sentences and chapter markers."""
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
    matcher = TextMatcher(audio_words, ebook_sentences)
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

def process_all_chapters(audio_dir: str, output_path: str = None):
    """
    Process all MP3 chapters in a directory and save transcriptions with chapter info.
    Can also process a single audio file.
    
    Args:
        audio_dir: Directory containing MP3 chapter files, or path to single audio file
        output_path: Optional path to save JSON output
        
    Returns:
        List of chapter dicts with transcription info
    """
    import json
    from pathlib import Path
    import whisper
    from rich.progress import Progress, SpinnerColumn, TextColumn
    
    # handle single file case
    audio_path = Path(audio_dir)
    if audio_path.is_file():
        # if it's a single file, treat it as chapter 1
        mp3_files = [audio_path]
        is_single_file = True
    else:
        # get all mp3 files in directory
        mp3_files = sorted(audio_path.glob("*.mp3"))
        if not mp3_files:
            raise ValueError(f"No MP3 files found in {audio_dir}")
        is_single_file = False
    
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
        for i, mp3_path in enumerate(mp3_files, start=1):
            task = progress.add_task(f"[cyan]Processing audio for chapter {i}...", total=len(mp3_files))
            
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
                "number": i,
                "filename": mp3_path.name,
                "duration": duration,
                "word_count": len(words),
                "words": words
            })
            
            progress.update(task, advance=1)
    
    # save results if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump({
                "book": audio_path.name if audio_path.is_file() else audio_path.name,
                "total_chapters": len(chapters),
                "total_words": sum(c["word_count"] for c in chapters),
                "chapters": chapters
            }, f, indent=2)
    
    return chapters

def match_book(audio_data: Dict, ebook_sentences: List[str], chapter_markers: Dict[int, str], audio_json_path: str) -> Dict[int, List[Dict]]:
    """
    Match entire audiobook with ebook text at once.
    
    Args:
        audio_data: Dict containing chapter transcriptions
        ebook_sentences: List of all sentences from ebook
        chapter_markers: Dict mapping sentence indices to chapter markers
        audio_json_path: Path to the audio JSON file
        
    Returns:
        Dict mapping chapter numbers to alignment results
    """
    from .matcher import TextMatcher
    from .audio import AudioProcessor
    from pathlib import Path
    
    # concatenate all audio words and adjust timestamps
    all_audio_words = []
    time_offset = 0.0
    
    # first pass: calculate total duration of each chapter
    chapter_durations = {}
    for chapter in audio_data["chapters"]:
        audio_path = Path(audio_json_path).parent / chapter["filename"]
        processor = AudioProcessor(str(audio_path))
        features = processor.process_chapter()
        chapter_durations[chapter["number"]] = features.duration
    
    # second pass: concatenate words with adjusted timestamps
    for chapter in audio_data["chapters"]:
        for word in chapter["words"]:
            adjusted_word = word.copy()
            adjusted_word["start"] += time_offset
            adjusted_word["end"] += time_offset
            all_audio_words.append(adjusted_word)
        time_offset += chapter_durations[chapter["number"]]
    
    # get silent regions for the entire book
    all_silent_regions = []
    time_offset = 0.0
    for chapter in audio_data["chapters"]:
        audio_path = Path(audio_json_path).parent / chapter["filename"]
        processor = AudioProcessor(str(audio_path))
        features = processor.process_chapter()
        for start, end in features.silent_regions:
            all_silent_regions.append((start + time_offset, end + time_offset))
        time_offset += features.duration
    
    # create matcher and match entire book
    matcher = TextMatcher(all_audio_words, ebook_sentences)
    all_results = matcher.match(all_audio_words, ebook_sentences, all_silent_regions)
    
    # split results back into chapters
    chapter_results = {}
    current_chapter = 0
    current_results = []
    
    # find chapter boundaries in ebook
    chapter_boundaries = {}
    for idx, marker in chapter_markers.items():
        if "chapter" in marker.lower() or "letter" in marker.lower():
            # extract chapter number from marker
            words = marker.lower().split()
            if "chapter" in words:
                num_idx = words.index("chapter") + 1
            else:
                num_idx = words.index("letter") + 1
            try:
                chapter_num = int(words[num_idx])
                chapter_boundaries[chapter_num] = idx
            except (ValueError, IndexError):
                continue
    
    # sort chapter boundaries
    sorted_chapters = sorted(chapter_boundaries.keys())
    
    # assign results to chapters based on sentence indices
    for result in all_results:
        # find which chapter this result belongs to
        result_chapter = 0
        for i in range(len(sorted_chapters) - 1):
            if result.sentence_idx >= chapter_boundaries[sorted_chapters[i]] and \
               result.sentence_idx < chapter_boundaries[sorted_chapters[i + 1]]:
                result_chapter = sorted_chapters[i]
                break
        else:
            # if not found in any chapter, assign to last chapter
            result_chapter = sorted_chapters[-1]
        
        # add to appropriate chapter's results
        if result_chapter not in chapter_results:
            chapter_results[result_chapter] = []
        chapter_results[result_chapter].append({
            'sentence': result.sentence,
            'start': result.start_time,
            'end': result.end_time,
            'confidence': result.confidence,
            'matched_text': result.matched_text,
            'is_silence_based': result.is_silence_based
        })
    
    return chapter_results

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
    
    # load audio transcriptions
    with open(audio_json) as f:
        audio_data = json.load(f)
    
    # parse ebook
    ebook_sentences, chapter_markers = parse_epub(ebook_path)
    
    # create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # match entire book at once
    chapter_results = match_book(audio_data, ebook_sentences, chapter_markers, audio_json)
    
    # save results for each chapter
    for chapter_num, results in chapter_results.items():
        output_path = output_dir / f"chapter_{chapter_num}_alignment.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2) 