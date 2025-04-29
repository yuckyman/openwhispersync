import json
from pathlib import Path
from typing import List, Tuple, Dict
import whisper
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import logging
from .ebook import parse_epub
from .audio import AudioProcessor
from .matcher import TextMatcher, SilenceRegion

console = Console()
logger = logging.getLogger(__name__)

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
              audio_path: str = None,
              silent_regions: List[Tuple[float, float]] = None) -> List[Dict]:
    """
    Match transcribed audio with ebook text.
    
    Args:
        audio_words: List of word dicts with 'text', 'start', 'end'
        ebook_sentences: List of sentences from ebook
        audio_path: Optional path to audio file for silence detection
        silent_regions: Optional list of (start, end) silence timestamps from transcription
        
    Returns:
        List of alignment dicts with 'sentence', 'start', 'end', 'confidence'
    """
    # use provided silent regions or get them from audio if path provided
    if silent_regions is None and audio_path:
        processor = AudioProcessor(audio_path)
        features = processor.process_chapter()
        silent_regions = features.silent_regions
    
    # create matcher and get results
    matcher = TextMatcher()  # Just configure with defaults
    results = matcher.match(audio_words, ebook_sentences, silent_regions)
    
    # convert to dict format
    return [
        {
            'sentence': r.sentence,
            'start_time': r.start_time,
            'end_time': r.end_time,
            'confidence': r.confidence,
            'matched_text': r.matched_text,
            'is_silence_based': r.is_silence_based,
            'punctuation_score': r.punctuation_score
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
            
            # get audio features first
            processor = AudioProcessor(str(mp3_path))
            features = processor.process_chapter()
            
            # Categorize silent regions by duration
            categorized_silences = []
            for start, end in features.silent_regions:
                duration = end - start
                if duration < 0.4:
                    silence_type = "brief"  # Potential commas, minor breaks
                elif duration < 1.0:
                    silence_type = "medium"  # Potential sentence boundaries
                else:
                    silence_type = "long"  # Paragraph breaks, chapter transitions
                
                categorized_silences.append({
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "type": silence_type
                })
            
            # then do whisper transcription
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
            
            # store everything we need
            chapters.append({
                "number": i,
                "filename": mp3_path.name,
                "duration": features.duration,
                "word_count": len(words),
                "words": words,
                "silent_regions": features.silent_regions,
                "categorized_silences": categorized_silences
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

def match_chapters(audio_json: str, ebook_path: str, output_dir: str):
    """
    Match transcribed audio chapters with ebook text, processing chapter by chapter.

    Args:
        audio_json: Path to JSON file with audio transcriptions
        ebook_path: Path to ebook file
        output_dir: Directory to save alignment results
    """
    import json
    from pathlib import Path
    from .ebook import parse_epub
    from .matcher import TextMatcher
    from rich.progress import Progress

    console.print(f"Loading audio transcriptions from [bold]{audio_json}[/bold]...")
    with open(audio_json) as f:
        audio_data = json.load(f)
    
    console.print(f"Parsing ebook [bold]{ebook_path}[/bold]...")
    ebook_sentences, chapter_markers = parse_epub(ebook_path)
    num_sentences = len(ebook_sentences)
    console.print(f"Ebook parsed: {num_sentences} sentences, {len(chapter_markers)} markers.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a single matcher instance to reuse
    matcher = TextMatcher() 

    # --- Determine Chapter Boundaries --- 
    # Map sentence index to (marker_text, chapter_number)
    # Chapter number might be None if it's just a marker like 'Title Page'
    # Use only markers that have a valid chapter number associated
    chapter_start_indices = sorted([
        (idx, data[1]) 
        for idx, data in chapter_markers.items() 
        if data[1] is not None # Ensure chapter number exists
    ])

    if not chapter_start_indices:
        logger.warning("No valid chapter markers found in epub. Treating entire book as Chapter 1.")
        # Assign chapter 1 to start at sentence 0
        chapter_start_indices = [(0, 1)]
    
    num_audio_chapters = len(audio_data.get("chapters", []))
    console.print(f"Matching {num_audio_chapters} audio chapters against {len(chapter_start_indices)} text chapters...")

    # --- Process Chapter by Chapter --- 
    with Progress() as progress:
        main_task = progress.add_task("[green]Aligning chapters...", total=len(chapter_start_indices))

        for i, (start_sentence_idx, chapter_num) in enumerate(chapter_start_indices):
            # Find corresponding audio chapter data
            # Assuming audio chapter numbers match text chapter numbers (1-based)
            audio_chapter_data = next((ch for ch in audio_data.get("chapters", []) if ch.get("number") == chapter_num), None)

            if not audio_chapter_data:
                logger.warning(f"No audio data found for Chapter {chapter_num}. Skipping.")
                progress.update(main_task, advance=1)
                continue

            # Define end sentence index for this chapter
            if i + 1 < len(chapter_start_indices):
                end_sentence_idx = chapter_start_indices[i+1][0]
            else:
                end_sentence_idx = num_sentences # Last chapter goes to the end
            
            # --- DEBUGGING: Print calculated slice indices --- 
            logger.debug(f"Chapter {chapter_num}: Using sentence slice indices {start_sentence_idx} to {end_sentence_idx}")
            
            # Slice the ebook sentences for this chapter
            chapter_sentences = ebook_sentences[start_sentence_idx:end_sentence_idx]

            # --- DEBUGGING: Print actual slice length --- 
            logger.debug(f"Chapter {chapter_num}: Actual slice length = {len(chapter_sentences)}")

            if not chapter_sentences:
                logger.warning(f"No ebook sentences found for Chapter {chapter_num} (Indices {start_sentence_idx}-{end_sentence_idx}). Skipping.")
                progress.update(main_task, advance=1)
                continue

            # Get audio words and silent regions for this chapter
            audio_words = audio_chapter_data.get("words", [])
            silent_regions = audio_chapter_data.get("silent_regions")
            categorized_silences = audio_chapter_data.get("categorized_silences")
            
            # If categorized_silences is missing, generate it from silent_regions
            if not categorized_silences and silent_regions:
                categorized_silences = []
                for start, end in silent_regions:
                    duration = end - start
                    if duration < 0.4:
                        silence_type = "brief"
                    elif duration < 1.0:
                        silence_type = "medium"
                    else:
                        silence_type = "long"
                    
                    categorized_silences.append({
                        "start": start,
                        "end": end,
                        "duration": duration,
                        "type": silence_type
                    })

            if not audio_words:
                logger.warning(f"No audio words found for Chapter {chapter_num}. Skipping.")
                progress.update(main_task, advance=1)
                continue
            
            console.print(f"  Matching Chapter {chapter_num} ({len(audio_words)} words vs {len(chapter_sentences)} sentences)...")
            
            # Run the matcher for this chapter
            # Matcher returns results with sentence_idx relative to chapter_sentences
            try:
                chapter_match_results = matcher.match(
                    audio_words,
                    chapter_sentences, # Pass only the slice
                    silent_regions
                )
            except Exception as e:
                logger.error(f"Error matching Chapter {chapter_num}: {e}", exc_info=True)
                progress.update(main_task, advance=1)
                continue

            # Adjust sentence indices to be absolute and format results
            final_results = []
            for r in chapter_match_results:
                final_results.append({
                    'sentence': r.sentence,
                    'sentence_idx': r.sentence_idx + start_sentence_idx, # Adjust index
                    'start_time': r.start_time,
                    'end_time': r.end_time,
                    'confidence': r.confidence,
                    'matched_text': r.matched_text,
                    'is_silence_based': r.is_silence_based,
                    'punctuation_score': r.punctuation_score
                })

            # Save results for this chapter
            output_path = output_dir / f"chapter_{chapter_num}_alignment.json"
            console.print(f"  Saving alignment for Chapter {chapter_num} to [bold]{output_path}[/bold]")
            with open(output_path, 'w') as f:
                json.dump(final_results, f, indent=2)
            
            progress.update(main_task, advance=1)

    console.print(f"[green]✓[/green] Chapter alignment complete. Results saved to [bold]{output_dir}[/bold]") 