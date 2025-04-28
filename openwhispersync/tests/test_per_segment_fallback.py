import logging
from pathlib import Path
import whisper
from whisperless.audio import AudioProcessor
from whisperless.ebook import parse_epub
from whisperless.matcher import TextMatcher
import json

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_chapter1_matching():
    # paths
    chapter_path = Path("whisperless/files/frankenstein/frankenstein_01_shelley_64kb.mp3")
    epub_path = Path("whisperless/files/frankenstein/frankenstein.epub")
    output_path = Path("whisperless/files/frankenstein/chapter1_results.json")
    
    # process audio
    logger.info("Processing audio...")
    processor = AudioProcessor(chapter_path)
    features = processor.process_chapter()
    
    # process ebook
    logger.info("Processing ebook...")
    sentences = parse_epub(epub_path)
    
    # transcribe audio
    logger.info("Transcribing audio...")
    model = whisper.load_model("base")
    result = model.transcribe(
        str(chapter_path),
        word_timestamps=True,
        language="en"
    )
    
    # convert whisper segments to our format
    audio_words = []
    for segment in result["segments"]:
        for word in segment["words"]:
            audio_words.append({
                "text": word["word"].strip(),
                "start": word["start"],
                "end": word["end"]
            })
    
    # create matcher with per-segment fallback
    matcher = TextMatcher(
        min_confidence=0.7,
        window_size=5,
        silence_threshold=0.5
    )
    
    # match text
    logger.info("Matching text...")
    results = matcher.match(audio_words, sentences, features.silent_regions)
    
    # save results
    logger.info(f"Saving results to {output_path}")
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
    
    logger.info("Test complete!")

if __name__ == "__main__":
    test_chapter1_matching() 