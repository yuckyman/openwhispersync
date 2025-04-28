import logging
from pathlib import Path
from whisperless.matcher import TextMatcher, MatchResult
from whisperless.ebook import parse_epub
from whisperless.audio import AudioProcessor
import json

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_silence_based_matching():
    """Test matching with silence detection."""
    # sample audio words with timestamps
    audio_words = [
        {"text": "hello", "start": 0.0, "end": 0.5},
        {"text": "world", "start": 0.5, "end": 1.0},
        {"text": "this", "start": 2.0, "end": 2.3},  # gap here
        {"text": "is", "start": 2.3, "end": 2.5},
        {"text": "a", "start": 2.5, "end": 2.6},
        {"text": "test", "start": 2.6, "end": 3.0}
    ]
    
    # sample ebook sentences
    ebook_sentences = [
        "Hello world!",
        "This is a test.",
        "Another sentence."
    ]
    
    # silence regions (1-2s gap between sentences)
    silent_regions = [(1.0, 2.0)]
    
    # create matcher and test
    matcher = TextMatcher(min_confidence=0.7)
    results = matcher.match(audio_words, ebook_sentences, silent_regions)
    
    # verify results
    assert len(results) == 2, "Should match two sentences"
    assert results[0].sentence == "Hello world!", "First sentence match"
    assert results[1].sentence == "This is a test.", "Second sentence match"
    assert results[0].is_silence_based, "First match should be silence-based"
    assert results[1].is_silence_based, "Second match should be silence-based"
    assert results[0].sentence_idx == 0, "First match should be at index 0"
    assert results[1].sentence_idx == 1, "Second match should be at index 1"
    
    logger.info("Silence-based matching test passed!")

def test_fallback_matching():
    """Test fallback to sliding window when silence detection fails."""
    # sample audio words with timestamps
    audio_words = [
        {"text": "hello", "start": 0.0, "end": 0.5},
        {"text": "world", "start": 0.5, "end": 1.0},
        {"text": "this", "start": 1.0, "end": 1.3},  # no gap
        {"text": "is", "start": 1.3, "end": 1.5},
        {"text": "a", "start": 1.5, "end": 1.6},
        {"text": "test", "start": 1.6, "end": 2.0}
    ]
    
    # sample ebook sentences
    ebook_sentences = [
        "Hello world!",
        "This is a test.",
        "Another sentence."
    ]
    
    # create matcher and test without silence regions
    matcher = TextMatcher(min_confidence=0.7)
    results = matcher.match(audio_words, ebook_sentences)
    
    # verify results
    assert len(results) == 2, "Should match two sentences"
    assert results[0].sentence == "Hello world!", "First sentence match"
    assert results[1].sentence == "This is a test.", "Second sentence match"
    assert not results[0].is_silence_based, "First match should not be silence-based"
    assert not results[1].is_silence_based, "Second match should not be silence-based"
    assert results[0].sentence_idx == 0, "First match should be at index 0"
    assert results[1].sentence_idx == 1, "Second match should be at index 1"
    
    logger.info("Fallback matching test passed!")

def test_confidence_threshold():
    """Test that low confidence matches are rejected."""
    # sample audio words with timestamps
    audio_words = [
        {"text": "hello", "start": 0.0, "end": 0.5},
        {"text": "world", "start": 0.5, "end": 1.0}
    ]
    
    # sample ebook sentences with one that won't match
    ebook_sentences = [
        "Hello world!",
        "This won't match at all"
    ]
    
    # create matcher with high confidence threshold
    matcher = TextMatcher(min_confidence=0.9)
    results = matcher.match(audio_words, ebook_sentences)
    
    # verify results
    assert len(results) == 1, "Should only match one sentence"
    assert results[0].sentence == "Hello world!", "Should match only high confidence sentence"
    assert results[0].sentence_idx == 0, "Match should be at index 0"
    
    logger.info("Confidence threshold test passed!")

def test_frankenstein_matching():
    """Test matching with actual frankenstein ebook and audio."""
    # paths
    ebook_path = Path("whisperless/files/frankenstein/frankenstein.epub")
    audio_path = Path("whisperless/files/frankenstein/frankenstein_01_shelley_64kb.mp3")
    
    # load ebook sentences
    logger.info("Loading frankenstein ebook...")
    ebook_sentences = parse_epub(str(ebook_path))
    logger.info(f"Loaded {len(ebook_sentences)} sentences from ebook")
    
    # process audio
    logger.info("Processing frankenstein audio...")
    processor = AudioProcessor(audio_path)
    features = processor.process_chapter()
    logger.info(f"Found {len(features.silent_regions)} silent regions")
    
    # TODO: load whisper transcription
    # for now, we'll use a sample of the first few sentences
    audio_words = [
        {"text": "i", "start": 0.0, "end": 0.1},
        {"text": "am", "start": 0.1, "end": 0.2},
        {"text": "by", "start": 0.2, "end": 0.3},
        {"text": "birth", "start": 0.3, "end": 0.4},
        {"text": "a", "start": 0.4, "end": 0.5},
        {"text": "genevese", "start": 0.5, "end": 0.7},
        {"text": "and", "start": 0.7, "end": 0.8},
        {"text": "my", "start": 0.8, "end": 0.9},
        {"text": "family", "start": 0.9, "end": 1.1},
        {"text": "is", "start": 1.1, "end": 1.2},
        {"text": "one", "start": 1.2, "end": 1.3},
        {"text": "of", "start": 1.3, "end": 1.4},
        {"text": "the", "start": 1.4, "end": 1.5},
        {"text": "most", "start": 1.5, "end": 1.6},
        {"text": "distinguished", "start": 1.6, "end": 1.8},
        {"text": "of", "start": 1.8, "end": 1.9},
        {"text": "that", "start": 1.9, "end": 2.0},
        {"text": "republic", "start": 2.0, "end": 2.2}
    ]
    
    # create matcher with lower confidence threshold
    matcher = TextMatcher(min_confidence=0.5, silence_threshold=0.1)  # lower silence threshold
    
    # add some silence regions
    silent_regions = [(0.0, 0.5), (2.2, 2.7)]  # longer pauses at start and end
    
    # match the audio words with ebook sentences
    results = matcher.match(audio_words, ebook_sentences, silent_regions)
    
    # debug logging
    logger.info(f"Found {len(results)} matches")
    for result in results:
        logger.info(f"Match: {result.matched_text} (confidence: {result.confidence:.2f})")
    
    # verify results
    assert len(results) > 0, "Should find at least one match"
    
    # log first few matches
    for i, result in enumerate(results[:3]):
        logger.info(f"Match {i+1}:")
        logger.info(f"  Sentence: {result.sentence}")
        logger.info(f"  Time: {result.start_time:.2f}s - {result.end_time:.2f}s")
        logger.info(f"  Confidence: {result.confidence:.2f}")
        logger.info(f"  Silence-based: {result.is_silence_based}")
        logger.info(f"  Matched text: {result.matched_text}")
    
    # save results for inspection
    output_path = Path("whisperless/files/frankenstein/matches.json")
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
    logger.info(f"Saved matches to {output_path}")

if __name__ == "__main__":
    test_silence_based_matching()
    test_fallback_matching()
    test_confidence_threshold()
    test_frankenstein_matching()
    logger.info("All tests passed!") 