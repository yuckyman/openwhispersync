import re
import math
import logging
from typing import List, TypedDict, Optional, Tuple, Dict
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

class SilenceRegion(TypedDict):
    start: float
    end: float
    duration: float
    type: str  # "brief", "medium", or "long"

@dataclass
class MatchResult:
    sentence: str
    sentence_idx: int
    start_time: float
    end_time: float
    confidence: float
    matched_text: str
    is_silence_based: bool = False
    punctuation_score: float = 0.0

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

class PunctuationAnalyzer:
    """Analyzes text for punctuation patterns and correlates with silence regions."""
    
    def __init__(self):
        # Punctuation patterns
        self.period_pattern = re.compile(r'[.!?][\s"]*$')
        self.comma_pattern = re.compile(r'[,;:][\s"]*$')
        self.paragraph_pattern = re.compile(r'[.!?][\s"]*$')  # For now, same as period
        
        # Score adjustments for punctuation-silence alignment
        self.period_silence_boost = 0.15
        self.comma_silence_boost = 0.10
        self.paragraph_silence_boost = 0.20
        self.missing_pause_penalty = 0.05
    
    def categorize_silence_regions(self, silent_regions: List[Tuple[float, float]]) -> List[SilenceRegion]:
        """Categorize silence regions by duration."""
        categorized = []
        
        for start, end in silent_regions:
            duration = end - start
            
            # Categorize by duration
            if duration < 0.4:
                silence_type = "brief"  # Potential commas, minor breaks
            elif duration < 1.0:
                silence_type = "medium"  # Potential sentence boundaries
            else:
                silence_type = "long"  # Paragraph breaks, chapter transitions
            
            categorized.append({
                "start": start,
                "end": end,
                "duration": duration,
                "type": silence_type
            })
        
        return categorized
    
    def analyze_sentence(self, sentence: str) -> Dict[str, bool]:
        """Analyze a sentence for punctuation patterns."""
        return {
            "ends_with_period": bool(self.period_pattern.search(sentence)),
            "has_commas": bool(self.comma_pattern.search(sentence)),
            "is_paragraph_end": bool(self.paragraph_pattern.search(sentence))
        }
    
    def calculate_punctuation_score(self, 
                                   sentence: str, 
                                   start_time: float, 
                                   end_time: float, 
                                   silent_regions: List[SilenceRegion]) -> float:
        """Calculate score adjustment based on punctuation-silence alignment."""
        analysis = self.analyze_sentence(sentence)
        score_adjustment = 0.0
        
        # Find silence regions that overlap with the sentence time span
        relevant_silences = [
            s for s in silent_regions 
            if (s["start"] >= start_time and s["start"] <= end_time) or
               (s["end"] >= start_time and s["end"] <= end_time)
        ]
        
        # Check for sentence ending alignment with medium/long silence
        if analysis["ends_with_period"]:
            # Look for medium or long silence near the end time
            end_aligned_silence = next(
                (s for s in relevant_silences 
                 if abs(s["start"] - end_time) < 0.3 
                 and s["type"] in ["medium", "long"]), 
                None
            )
            
            if end_aligned_silence:
                score_adjustment += self.period_silence_boost
            else:
                # Penalty for period with no corresponding pause
                score_adjustment -= self.missing_pause_penalty
        
        # Check for comma alignment with brief silence
        if analysis["has_commas"]:
            # Look for brief silences within the sentence
            comma_aligned_silences = [
                s for s in relevant_silences 
                if s["type"] == "brief"
            ]
            
            if comma_aligned_silences:
                # Boost for each aligned comma/silence (up to 2)
                score_adjustment += min(len(comma_aligned_silences), 2) * self.comma_silence_boost
        
        # Check for paragraph ending alignment with long silence
        if analysis["is_paragraph_end"]:
            # Look for long silence near the end time
            paragraph_aligned_silence = next(
                (s for s in relevant_silences 
                 if abs(s["start"] - end_time) < 0.5 
                 and s["type"] == "long"), 
                None
            )
            
            if paragraph_aligned_silence:
                score_adjustment += self.paragraph_silence_boost
        
        return score_adjustment

class TextMatcher:
    def __init__(self, min_conf: float = 0.7):
        """Initialize matcher with just configuration parameters."""
        self.min_conf = min_conf
        self.punctuation_analyzer = PunctuationAnalyzer()

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
        
        # categorize silence regions if available
        categorized_silences = []
        if silent_regions:
            categorized_silences = self.punctuation_analyzer.categorize_silence_regions(silent_regions)
        
        return self._run_matching(
            audio_words=audio_words,
            sentences=ebook_sentences,
            silent_regions=silent_regions or [],
            categorized_silences=categorized_silences
        )

    def _run_matching(self, 
                     audio_words: List[AudioWord], 
                     sentences: List[str], 
                     silent_regions: List[Tuple[float, float]] = None,
                     categorized_silences: List[SilenceRegion] = None) -> List[MatchResult]:
        """Run the matching algorithm."""
        results = []
        total_words = len(audio_words)
        
        # use console.log instead of progress bar to avoid conflicts
        console = Console()
        console.log(f"Processing {total_words} words...")
        
        # if we have silent regions, try silence-based matching first
        if silent_regions:
            segments = self._get_silence_based_segments(audio_words, silent_regions, categorized_silences)
            silence_matches = 0
            for segment in segments:
                if match := self._match_segment(segment, sentences, categorized_silences):
                    results.append(match)
                    silence_matches += 1
            
            if silence_matches > 0:
                console.log(f"Found {silence_matches} matches using silence-based segmentation")
                return results
        
        # fallback to sliding window approach with dynamic sizing
        console.log("Falling back to sliding window approach...")
        
        last_match_idx = 0  # Track last matched sentence for adaptive window
        i = 0
        
        while i < len(audio_words) - 5:  # minimum window size of 5
            # Dynamic window sizing based on position and previous matches
            window_size = self._calculate_window_size(i, audio_words, sentences, last_match_idx)
            
            if i % 1000 == 0:  # log progress every 1000 words
                console.log(f"Processing word {i} of {total_words}... (window size: {window_size})")
            
            # Ensure we don't go past the end of audio_words
            window_end = min(i + window_size, len(audio_words))
            window = audio_words[i:window_end]
            
            if match := self._match_window(window, sentences, categorized_silences):
                results.append(match)
                last_match_idx = match.sentence_idx
                # Skip ahead to avoid overlapping matches
                i += max(5, window_size // 2)
            else:
                # Move forward by 1 word if no match
                i += 1
        
        console.log(f"Found {len(results)} total matches")
        return results

    def _calculate_window_size(self, 
                              pos: int, 
                              audio_words: List[AudioWord], 
                              sentences: List[str],
                              last_match_idx: int) -> int:
        """Calculate dynamic window size based on context."""
        # Base window size
        base_size = 10
        
        # Adjust window size based on sentence complexity
        if last_match_idx < len(sentences):
            # Look at the next few sentences for punctuation density
            next_sentences = sentences[last_match_idx:min(last_match_idx + 3, len(sentences))]
            punctuation_count = sum(s.count(',') + s.count('.') + s.count('!') + s.count('?') 
                                   for s in next_sentences)
            
            # More punctuation = smaller window for precision
            if punctuation_count > 10:
                base_size = 8  # Smaller window for heavily punctuated text
            elif punctuation_count < 3:
                base_size = 12  # Larger window for sparse punctuation
        
        # Ensure minimum and maximum bounds
        return max(5, min(base_size, 15))

    def _get_silence_based_segments(self, 
                                   audio_words: List[AudioWord], 
                                   silent_regions: List[Tuple[float, float]],
                                   categorized_silences: List[SilenceRegion] = None) -> List[List[AudioWord]]:
        """Get segments based on silence regions with punctuation awareness."""
        segments = []
        
        # Focus on medium and long silences for segmentation
        significant_silences = []
        if categorized_silences:
            significant_silences = [s for s in categorized_silences 
                                  if s["type"] in ["medium", "long"]]
        else:
            # If not categorized, use all silences longer than 0.4s
            significant_silences = [{"start": start, "end": end}
                                  for start, end in silent_regions
                                  if end - start > 0.4]
        
        for silence in significant_silences:
            start, end = silence["start"], silence["end"]
            
            # Find words that occur right before silence
            pre_silence_words = [w for w in audio_words 
                               if w['end'] <= start and w['end'] > start - 2.0]
            
            # Find words right after silence
            post_silence_words = [w for w in audio_words 
                                if w['start'] >= end and w['start'] < end + 2.0]
            
            if pre_silence_words and post_silence_words:
                # Create segment around silence with context words
                segment_words = pre_silence_words[-5:] + post_silence_words[:5]
                segments.append(segment_words)
        
        return segments

    def _match_segment(self, 
                      segment: List[AudioWord], 
                      sentences: List[str],
                      categorized_silences: List[SilenceRegion] = None) -> Optional[MatchResult]:
        """Match a segment of audio words against the text sentences with punctuation awareness."""
        # Get segment text
        segment_text = ' '.join(w['text'] for w in segment)
        segment_text = clean_for_matching(segment_text)
        
        # Get segment timestamps
        start_time = segment[0]['start']
        end_time = segment[-1]['end']
        
        # Try to match against each sentence
        best_match = None
        best_score = 0
        best_idx = -1
        best_punct_score = 0
        
        for i, sentence in enumerate(sentences):
            # Clean sentence
            clean_sent = clean_for_matching(sentence)
            
            # Compute base match score
            score = fuzz.token_sort_ratio(segment_text, clean_sent)
            
            # Calculate punctuation score adjustment if silences are categorized
            punct_score = 0
            if categorized_silences:
                punct_score = self.punctuation_analyzer.calculate_punctuation_score(
                    sentence, start_time, end_time, categorized_silences
                )
            
            # Adjust score with punctuation alignment
            adjusted_score = score + (punct_score * 100)
            
            # Update best match if score is better
            if adjusted_score > best_score and adjusted_score >= self.min_conf * 100:
                best_score = adjusted_score
                best_match = sentence
                best_idx = i
                best_punct_score = punct_score
        
        # Return result if we found a match
        if best_match is not None:
            return MatchResult(
                sentence=best_match,
                sentence_idx=best_idx,
                start_time=start_time,
                end_time=end_time,
                confidence=best_score / 100,
                matched_text=segment_text,
                is_silence_based=True,
                punctuation_score=best_punct_score
            )
        
        return None

    def _match_window(self, 
                     window: List[AudioWord], 
                     sentences: List[str],
                     categorized_silences: List[SilenceRegion] = None) -> Optional[MatchResult]:
        """Match a window of audio words against the text sentences with punctuation awareness."""
        # Get window text
        window_text = ' '.join(w['text'] for w in window)
        window_text = clean_for_matching(window_text)
        
        # Get window timestamps
        start_time = window[0]['start']
        end_time = window[-1]['end']
        
        # Try to match against each sentence
        best_match = None
        best_score = 0
        best_idx = -1
        best_punct_score = 0
        
        for i, sentence in enumerate(sentences):
            # Clean sentence
            clean_sent = clean_for_matching(sentence)
            
            # Compute base match score
            score = fuzz.token_sort_ratio(window_text, clean_sent)
            
            # Calculate punctuation score adjustment if silences are categorized
            punct_score = 0
            if categorized_silences:
                punct_score = self.punctuation_analyzer.calculate_punctuation_score(
                    sentence, start_time, end_time, categorized_silences
                )
            
            # Adjust score with punctuation alignment
            adjusted_score = score + (punct_score * 100)
            
            # Update best match if score is better
            if adjusted_score > best_score and adjusted_score >= self.min_conf * 100:
                best_score = adjusted_score
                best_match = sentence
                best_idx = i
                best_punct_score = punct_score
        
        # Return result if we found a match
        if best_match is not None:
            return MatchResult(
                sentence=best_match,
                sentence_idx=best_idx,
                start_time=start_time,
                end_time=end_time,
                confidence=best_score / 100,
                matched_text=window_text,
                is_silence_based=False,
                punctuation_score=best_punct_score
            )
        
        return None