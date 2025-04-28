from typing import List, Tuple, Dict, Optional
import numpy as np
from rapidfuzz import fuzz, process
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class MatchResult:
    """Container for fuzzy matching results."""
    sentence: str
    start_time: float
    end_time: float
    confidence: float
    matched_text: str
    is_silence_based: bool
    sentence_idx: int  # index of the matched sentence in the ebook

class TextMatcher:
    """Handles fuzzy matching between transcribed audio and ebook text."""
    
    def __init__(self, 
                 min_confidence: float = 0.6,
                 window_size: int = 5,
                 silence_threshold: float = 0.5):
        """
        Args:
            min_confidence: Minimum confidence score for a match (0-1)
            window_size: Number of sentences to consider in sliding window
            silence_threshold: Minimum silence duration to consider a boundary (seconds)
        """
        self.min_confidence = min_confidence
        self.window_size = window_size
        self.silence_threshold = silence_threshold
    
    def _preprocess_text(self, text: str) -> str:
        """Clean text for comparison."""
        # lowercase and remove punctuation
        text = text.lower()
        text = ''.join(c for c in text if c.isalnum() or c.isspace())
        return text
    
    def _find_best_match(self, 
                        target: str, 
                        candidates: List[str],
                        start_idx: int = 0) -> Tuple[int, float, str]:
        """Find best matching sentence using rapidfuzz."""
        processed_target = self._preprocess_text(target)
        processed_candidates = [self._preprocess_text(c) for c in candidates]
        
        logger.debug(f"Finding best match for: '{processed_target}'")
        logger.debug(f"Candidates: {processed_candidates}")
        
        # use rapidfuzz's process.extractOne for efficient matching
        best_match = process.extractOne(
            processed_target,
            processed_candidates,
            scorer=fuzz.ratio,
            score_cutoff=self.min_confidence * 100
        )
        
        if best_match:
            match_text, score, idx = best_match
            logger.debug(f"Found match: '{match_text}' with score {score}")
            return start_idx + idx, score / 100, candidates[idx]
        
        logger.debug("No match found above confidence threshold")
        return -1, 0.0, ""
    
    def _get_silence_based_segments(self,
                                  audio_words: List[Dict],
                                  silent_regions: List[Tuple[float, float]]) -> List[Dict]:
        """Group words into segments based on silence detection."""
        segments = []
        current_segment = []
        current_start = audio_words[0]['start']
        
        # filter and merge silence regions
        valid_silences = []
        if silent_regions:
            # sort by start time
            silent_regions.sort(key=lambda x: x[0])
            
            # merge nearby silence regions (within 0.5s of each other)
            current_start, current_end = silent_regions[0]
            for start, end in silent_regions[1:]:
                if start - current_end <= 0.5:  # merge if gap is small
                    current_end = max(current_end, end)
                else:
                    if current_end - current_start >= self.silence_threshold:
                        valid_silences.append((current_start, current_end))
                    current_start, current_end = start, end
            
            # add the last region
            if current_end - current_start >= self.silence_threshold:
                valid_silences.append((current_start, current_end))
        
        logger.info(f"Found {len(valid_silences)} valid silence regions after merging:")
        for i, (start, end) in enumerate(valid_silences):
            logger.info(f"  Silence {i+1}: {start:.2f}s - {end:.2f}s")
        
        # keep track of which silence regions we've used
        used_silences = set()
        
        # minimum duration for a valid speech segment (in seconds)
        min_segment_duration = 1.0
        
        for word in audio_words:
            word_start = word['start']
            word_end = word['end']
            
            # check if we're at the start of a new silence region
            for i, (start, end) in enumerate(valid_silences):
                if i not in used_silences and word_start >= start and current_segment:
                    # only end segment if it's long enough
                    segment_duration = current_segment[-1]['end'] - current_start
                    if segment_duration >= min_segment_duration:
                        segment_text = ' '.join(w['text'] for w in current_segment)
                        segments.append({
                            'text': segment_text,
                            'start': current_start,
                            'end': current_segment[-1]['end']
                        })
                        logger.info(f"Created segment: '{segment_text}' ({current_start:.2f}s - {current_segment[-1]['end']:.2f}s)")
                        current_segment = []
                        current_start = word_start
                        used_silences.add(i)
                    break
            
            # only skip words that are completely within silence regions
            should_skip = False
            for start, end in valid_silences:
                if start <= word_start and word_end <= end:
                    should_skip = True
                    logger.debug(f"Skipping word during silence: {word['text']} at {word_start:.2f}s")
                    break
            
            if should_skip:
                continue
                
            current_segment.append(word)
        
        # add final segment if it's long enough
        if current_segment:
            segment_duration = current_segment[-1]['end'] - current_start
            if segment_duration >= min_segment_duration:
                segment_text = ' '.join(w['text'] for w in current_segment)
                segments.append({
                    'text': segment_text,
                    'start': current_start,
                    'end': current_segment[-1]['end']
                })
                logger.info(f"Created final segment: '{segment_text}' ({current_start:.2f}s - {current_segment[-1]['end']:.2f}s)")
        
        return segments
    
    def _word_convolution_match(self,
                              audio_words: List[Dict],
                              ebook_sentences: List[str],
                              window_size: int = 10,
                              search_window: int = 200) -> List[MatchResult]:
        """Match words using a sliding window convolution approach."""
        results = []
        
        # convert ebook sentences to words with better preprocessing
        ebook_words = []
        for sentence in ebook_sentences:
            # expand contractions and normalize text
            sentence = sentence.lower()
            sentence = sentence.replace("'", " ")  # handle contractions
            sentence = ''.join(c for c in sentence if c.isalnum() or c.isspace())
            words = sentence.split()
            ebook_words.extend(words)
        
        # convert audio words to just text with same preprocessing
        audio_text = []
        for word in audio_words:
            word = word['text'].lower()
            word = word.replace("'", " ")  # handle contractions
            word = ''.join(c for c in word if c.isalnum() or c.isspace())
            audio_text.append(word)
        
        logger.info(f"Ebook has {len(ebook_words)} words, audio has {len(audio_text)} words")
        
        # keep track of our position in both texts
        audio_pos = 0
        ebook_pos = 0
        
        while audio_pos < len(audio_text) - window_size + 1:
            # get current window from audio
            window = audio_text[audio_pos:audio_pos + window_size]
            
            # only search a local window in the ebook
            search_start = max(0, ebook_pos - search_window)
            search_end = min(len(ebook_words), ebook_pos + search_window)
            
            best_score = 0
            best_ebook_pos = ebook_pos
            
            # search in local window
            for j in range(search_start, search_end - window_size + 1):
                ebook_window = ebook_words[j:j + window_size]
                
                # calculate similarity score
                matches = sum(1 for a, b in zip(window, ebook_window) if a == b)
                score = matches / window_size
                
                if score > best_score:
                    best_score = score
                    best_ebook_pos = j
            
            if best_score > 0.6:  # require at least 60% match
                logger.info(f"Found match at audio pos {audio_pos}: {window} (score: {best_score:.2f})")
                
                # create match result
                start_time = audio_words[audio_pos]['start']
                end_time = audio_words[audio_pos + window_size - 1]['end']
                
                # find the sentence containing these words
                sentence_idx = 0
                for i, sentence in enumerate(ebook_sentences):
                    if all(word in sentence.lower() for word in window):
                        sentence_idx = i
                        break
                
                results.append(MatchResult(
                    sentence=ebook_sentences[sentence_idx],
                    start_time=start_time,
                    end_time=end_time,
                    confidence=best_score,
                    matched_text=' '.join(window),
                    is_silence_based=False,
                    sentence_idx=sentence_idx
                ))
                
                # update positions
                audio_pos += window_size
                ebook_pos = best_ebook_pos + window_size
            else:
                # no good match found, try next position
                audio_pos += 1
        
        return results
    
    def match(self,
              audio_words: List[Dict],
              ebook_sentences: List[str],
              silent_regions: Optional[List[Tuple[float, float]]] = None) -> List[MatchResult]:
        """
        Match transcribed audio segments with ebook sentences.
        
        Args:
            audio_words: List of word dicts with 'text', 'start', 'end'
            ebook_sentences: List of sentences from ebook
            silent_regions: Optional list of (start, end) silence timestamps
            
        Returns:
            List of MatchResult objects with alignment info
        """
        results = []
        
        # convert audio words to text
        audio_text = ' '.join(word['text'] for word in audio_words)
        logger.info(f"Processing audio text: {audio_text[:100]}...")
        
        # log first few ebook sentences for debugging
        logger.info("First few ebook sentences:")
        for i, sentence in enumerate(ebook_sentences[:10]):
            logger.info(f"  {i}: {sentence}")
        
        # find where the actual chapter content starts in the ebook
        ebook_start_idx = 0
        for i, sentence in enumerate(ebook_sentences):
            if "chapter" in sentence.lower() or "letter" in sentence.lower():
                ebook_start_idx = i + 1  # start after the chapter header
                logger.info(f"Found chapter start in ebook at index {i}: {sentence}")
                break
        
        if ebook_start_idx > 0:
            logger.info(f"Skipping {ebook_start_idx} sentences of metadata")
            ebook_sentences = ebook_sentences[ebook_start_idx:]
        
        # skip librivox intro by finding where the actual chapter starts
        chapter_start_idx = 0
        for i, word in enumerate(audio_words):
            if "chapter" in word['text'].lower():
                chapter_start_idx = i + 1  # start after the chapter header
                logger.info(f"Found chapter start in audio at word index {i}: {word['text']}")
                break
        
        if chapter_start_idx > 0:
            logger.info(f"Skipping {chapter_start_idx} words of librivox intro")
            audio_words = audio_words[chapter_start_idx:]
        
        # try word-level convolution matching with a smaller search window
        results = self._word_convolution_match(audio_words, ebook_sentences, search_window=50)
        
        if not results:
            logger.warning("No matches found with word convolution")
        
        return results 