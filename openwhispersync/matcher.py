import re
import math
import logging
from typing import List, TypedDict, Optional, Tuple
from dataclasses import dataclass
from rapidfuzz import fuzz, process
from rich.progress import Progress
from rich.console import Console

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AudioWord(TypedDict):
    text: str
    start: float
    end: float

@dataclass
class MatchResult:
    sentence: str
    sentence_idx: int
    start_time: float
    end_time: float
    confidence: float
    matched_text: str
    is_silence_based: bool = False

def clean_for_matching(text: str) -> str:
    """Clean text for fuzzy matching."""
    # convert to lowercase
    text = text.lower()
    
    # remove punctuation except apostrophes
    text = re.sub(r'[^\w\s\']', ' ', text)
    
    # normalize whitespace
    text = ' '.join(text.split())
    
    return text

class WordImportance:
    """Simple TF-based importance with stopword downweighting."""
    def __init__(self, sentences: List[str]):
        self.counts = {}
        self.total = 0
        self.stop = {'the','to','and','a','of','in','is','it','you','that'}
        for s in sentences:
            for w in clean_for_matching(s).split():
                self.total += 1
                if w not in self.stop:
                    self.counts[w] = self.counts.get(w, 0) + 1

    def score(self, w: str) -> float:
        w = clean_for_matching(w)
        if w in self.stop:
            return 0.1
        freq = self.counts.get(w, 1) / max(1, self.total)
        sc = 1 - math.log(freq + 1, 2)
        return max(0.2, min(1.0, sc))

class TextMatcher:
    def __init__(self, min_conf: float = 0.7):
        """Initialize matcher with just configuration parameters."""
        self.min_conf = min_conf

    def match(
        self,
        audio_words: List[AudioWord],
        ebook_sentences: List[str],
        silent_regions: Optional[List[Tuple[float, float]]] = None,
    ) -> List[MatchResult]:
        """Match audio words to ebook sentences without modifying instance state."""
        # prepare text for matching
        proc_sent = [clean_for_matching(s) for s in ebook_sentences]
        imp = WordImportance(ebook_sentences)
        
        return self._run_matching(
            audio_words=audio_words,
            sentences=ebook_sentences,
            silent_regions=silent_regions or []
        )

    def _run_matching(self, audio_words: List[AudioWord], sentences: List[str], silent_regions: List[Tuple[float, float]] = None) -> List[MatchResult]:
        """Run the matching algorithm."""
        results = []
        total_words = len(audio_words)
        
        # use console.log instead of progress bar to avoid conflicts
        console = Console()
        console.log(f"Processing {total_words} words...")
        
        # if we have silent regions, try silence-based matching first
        if silent_regions:
            segments = self._get_silence_based_segments(audio_words, silent_regions)
            silence_matches = 0
            for segment in segments:
                if match := self._match_segment(segment, sentences):
                    results.append(match)
                    silence_matches += 1
            
            if silence_matches > 0:
                console.log(f"Found {silence_matches} matches using silence-based segmentation")
                return results
        
        # fallback to sliding window approach
        window_size = 10  # reduced from 15 for better precision
        for i in range(0, len(audio_words) - window_size + 1):
            if i % 1000 == 0:  # log progress every 1000 words
                console.log(f"Processing word {i} of {total_words}...")
            
            window = audio_words[i:i + window_size]
            if match := self._match_window(window, sentences):
                results.append(match)
        
        console.log(f"Found {len(results)} total matches")
        return results

    def _get_silence_based_segments(self, audio_words: List[AudioWord], silent_regions: List[Tuple[float, float]]) -> List[List[AudioWord]]:
        segments = []
        for start, end in silent_regions:
            # find words that occur right before silence
            pre_silence_words = [w for w in audio_words 
                               if w['end'] <= start and w['end'] > start - 2.0]
            
            # find words right after silence
            post_silence_words = [w for w in audio_words 
                                if w['start'] >= end and w['start'] < end + 2.0]
            
            if pre_silence_words and post_silence_words:
                # create text windows around silence
                pre_text = ' '.join(w['text'] for w in pre_silence_words[-5:])
                post_text = ' '.join(w['text'] for w in post_silence_words[:5])
                
                # try to match these against sentence boundaries in the text
                for i, (sent1, sent2) in enumerate(zip(proc_sent[:-1], proc_sent[1:])):
                    # check if pre-silence matches end of sent1
                    # and post-silence matches start of sent2
                    if (fuzz.partial_ratio(clean_for_matching(pre_text), sent1[-50:]) > 80 and
                        fuzz.partial_ratio(clean_for_matching(post_text), sent2[:50]) > 80):
                        
                        segments.append(pre_silence_words + post_silence_words)
        return segments

    def _match_segment(self, segment: List[AudioWord], sentences: List[str]) -> Optional[MatchResult]:
        """Match a segment of audio words against the text sentences."""
        # get segment text
        segment_text = ' '.join(w['text'] for w in segment)
        segment_text = clean_for_matching(segment_text)
        
        # get segment timestamps
        start_time = segment[0]['start']
        end_time = segment[-1]['end']
        
        # try to match against each sentence
        best_match = None
        best_score = 0
        best_idx = -1
        
        for i, sentence in enumerate(sentences):
            # clean sentence
            clean_sent = clean_for_matching(sentence)
            
            # compute match score
            score = fuzz.token_sort_ratio(segment_text, clean_sent)
            
            # update best match if score is better
            if score > best_score and score >= self.min_conf * 100:
                best_score = score
                best_match = sentence
                best_idx = i
        
        # return result if we found a match
        if best_match is not None:
            return MatchResult(
                sentence=best_match,
                sentence_idx=best_idx,
                start_time=start_time,
                end_time=end_time,
                confidence=best_score / 100,
                matched_text=segment_text,
                is_silence_based=True
            )
        
        return None

    def _match_window(self, window: List[AudioWord], sentences: List[str]) -> Optional[MatchResult]:
        """Match a window of audio words against the text sentences."""
        # get window text
        window_text = ' '.join(w['text'] for w in window)
        window_text = clean_for_matching(window_text)
        
        # get window timestamps
        start_time = window[0]['start']
        end_time = window[-1]['end']
        
        # try to match against each sentence
        best_match = None
        best_score = 0
        best_idx = -1
        
        for i, sentence in enumerate(sentences):
            # clean sentence
            clean_sent = clean_for_matching(sentence)
            
            # compute match score
            score = fuzz.token_sort_ratio(window_text, clean_sent)
            
            # update best match if score is better
            if score > best_score and score >= self.min_conf * 100:
                best_score = score
                best_match = sentence
                best_idx = i
        
        # return result if we found a match
        if best_match is not None:
            return MatchResult(
                sentence=best_match,
                sentence_idx=best_idx,
                start_time=start_time,
                end_time=end_time,
                confidence=best_score / 100,
                matched_text=window_text,
                is_silence_based=False
            )
        
        return None