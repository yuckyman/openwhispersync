from typing import List, Tuple, Dict, Optional, TypedDict
import numpy as np
from rapidfuzz import fuzz, process
import logging
from dataclasses import dataclass
import math

class AudioWord(TypedDict):
    """Type definition for a word in the audio transcript."""
    text: str  # the word text
    start: float  # start time in seconds
    end: float  # end time in seconds

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

class WordImportance:
    """Handles word importance scoring based on frequency in the text."""
    
    def __init__(self, ebook_sentences: List[str]):
        """Initialize word importance calculator with ebook text.
        
        Args:
            ebook_sentences: List of sentences from the ebook
        """
        self.word_counts = {}
        self.total_words = 0
        self._calculate_word_frequencies(ebook_sentences)
        
    def _calculate_word_frequencies(self, sentences: List[str]):
        """Calculate word frequencies from the ebook text."""
        # common words that should have lower importance
        self.stopwords = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
            'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they',
            'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one',
            'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out',
            'if', 'about', 'who', 'get', 'which', 'go', 'me'
        }
        
        for sentence in sentences:
            # clean and split the sentence
            words = self._clean_text(sentence).split()
            self.total_words += len(words)
            
            # count word frequencies
            for word in words:
                if word not in self.stopwords:
                    self.word_counts[word] = self.word_counts.get(word, 0) + 1
    
    def _clean_text(self, text: str) -> str:
        """Clean text for word counting."""
        text = text.lower()
        # preserve contractions but remove other punctuation
        text = text.replace("'", "")
        return ''.join(c for c in text if c.isalnum() or c.isspace())
    
    def get_word_importance(self, word: str) -> float:
        """Calculate importance score for a word.
        
        Args:
            word: The word to score
            
        Returns:
            float: Importance score (0.0 to 1.0)
        """
        word = self._clean_text(word)
        
        # stopwords get minimal importance
        if word in self.stopwords:
            return 0.1
            
        # rare words get higher importance
        count = self.word_counts.get(word, 1)
        frequency = count / self.total_words
        
        # use log scale to dampen the effect
        importance = 1.0 - (math.log(frequency + 1) / math.log(2))
        return max(0.2, min(1.0, importance))
    
    def get_weighted_match_score(self, window: List[str], ebook_window: List[str]) -> float:
        """Calculate weighted match score between two word sequences.
        
        Args:
            window: List of words from audio
            ebook_window: List of words from ebook
            
        Returns:
            float: Weighted match score (0.0 to 1.0)
        """
        if not window or not ebook_window:
            return 0.0
            
        # get common words and their importance scores
        common_words = set(window).intersection(ebook_window)
        importance_sum = sum(self.get_word_importance(word) for word in common_words)
        
        # weight by position similarity
        position_penalty = 0.0
        for word in common_words:
            # find positions of word in both windows
            audio_pos = [i for i, w in enumerate(window) if w == word]
            ebook_pos = [i for i, w in enumerate(ebook_window) if w == word]
            
            if audio_pos and ebook_pos:
                # calculate relative position difference
                rel_audio_pos = audio_pos[0] / len(window)
                rel_ebook_pos = ebook_pos[0] / len(ebook_window)
                position_penalty += abs(rel_audio_pos - rel_ebook_pos)
        
        if common_words:
            position_penalty /= len(common_words)
            
        # combine importance and position scores
        base_score = importance_sum / max(len(window), len(ebook_window))
        return base_score * (1.0 - position_penalty * 0.5)  # position affects up to 50% of score

class TextMatcher:
    """Handles fuzzy matching between transcribed audio and ebook text."""
    
    def __init__(self, 
                 audio_words: List[AudioWord], 
                 ebook_sentences: List[str], 
                 silent_regions: Optional[List[Tuple[float, float]]] = None,
                 min_confidence: float = 0.5,  # lowered from 0.6
                 silence_threshold: float = 0.1,  # lowered from 0.5
                 max_words_per_segment: int = 20):
        """
        Initialize text matcher.
        
        Args:
            audio_words: List of word dicts with 'text', 'start', 'end'
            ebook_sentences: List of sentences from ebook
            silent_regions: Optional list of (start, end) silence timestamps
            min_confidence: Minimum confidence score for a match (0-1)
            silence_threshold: Minimum duration (seconds) for silence detection
            max_words_per_segment: Maximum number of words per segment
        """
        self.min_confidence = min_confidence
        self.silence_threshold = silence_threshold
        self.max_words_per_segment = max_words_per_segment
    
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
    
    def _detect_intro(self, audio_words: List[Dict], ebook_sentences: List[str], chapter_num: int) -> int:
        """Detect where the librivox intro ends and the chapter begins.
        
        Args:
            audio_words: List of word dicts with 'text', 'start', 'end'
            ebook_sentences: List of sentences from ebook
            chapter_num: Chapter number to find
            
        Returns:
            Index in audio_words where the chapter content begins
        """
        # first look for librivox intro phrases
        intro_phrases = [
            "librivox",
            "recording",
            "public domain",
            "read by",
            "chapter",
            "letter",
            "this is a librivox recording",
            "all librivox recordings are in the public domain",
            "for more information",
            "please visit",
            "librivox.org",
            "today's reading"
        ]
        
        # look for the last occurrence of any intro phrase
        last_intro_idx = -1
        for i, word in enumerate(audio_words):
            if any(phrase in word['text'].lower() for phrase in intro_phrases):
                last_intro_idx = i
        
        # if we found an intro, skip to after it
        if last_intro_idx >= 0:
            logger.info(f"Found librivox intro ending at word index {last_intro_idx}")
            # add a buffer of 5 words to make sure we're past the intro
            return last_intro_idx + 5
        
        # if no intro found, try to find the chapter header in the ebook
        chapter_start_sentence = None
        chapter_start_idx = 0
        
        # look for chapter markers with more flexibility
        chapter_markers = [
            f"chapter {chapter_num}",
            f"letter {chapter_num}",
            f"chapter {self._number_to_roman(chapter_num)}",
            f"letter {self._number_to_roman(chapter_num)}"
        ]
        
        for i, sentence in enumerate(ebook_sentences):
            if any(marker in sentence.lower() for marker in chapter_markers):
                # get the next few sentences as potential chapter start
                chapter_start_sentence = ' '.join(ebook_sentences[i+1:i+10]).lower()  # increased window to 10 sentences
                chapter_start_idx = i + 1
                break
        
        if not chapter_start_sentence:
            logger.warning(f"Could not find chapter {chapter_num} in ebook")
            return 0
            
        # clean up the chapter start text
        chapter_start_words = [w for w in chapter_start_sentence.split() if w.isalnum()]
        
        # look for these words in the audio with a sliding window
        best_match_idx = -1
        best_match_score = 0
        window_size = min(30, len(chapter_start_words))  # increased window size
        
        # look further ahead in the audio (first 200 words)
        search_range = min(200, len(audio_words))
        for i in range(search_range - window_size + 1):
            # get a window of words from audio
            window = [w['text'].lower() for w in audio_words[i:i+window_size]]
            
            # calculate match score using fuzzy matching
            score = fuzz.ratio(' '.join(window), ' '.join(chapter_start_words[:window_size])) / 100
            
            if score > best_match_score:
                best_match_score = score
                best_match_idx = i
        
        if best_match_score > 0.3:  # lowered threshold
            logger.info(f"Found chapter {chapter_num} start at word index {best_match_idx} (score: {best_match_score:.2f})")
            return best_match_idx
        
        return 0

    def _number_to_roman(self, num: int) -> str:
        """Convert number to roman numeral."""
        val = [
            1000, 900, 500, 400,
            100, 90, 50, 40,
            10, 9, 5, 4,
            1
        ]
        syb = [
            "M", "CM", "D", "CD",
            "C", "XC", "L", "XL",
            "X", "IX", "V", "IV",
            "I"
        ]
        roman_num = ''
        i = 0
        while num > 0:
            for _ in range(num // val[i]):
                roman_num += syb[i]
                num -= val[i]
            i += 1
        return roman_num

    def _word_convolution_match(self,
                              audio_words: List[Dict],
                              ebook_sentences: List[str],
                              window_size: int = 15,
                              search_window: int = 500) -> List[MatchResult]:
        """Match words using a sliding window convolution approach with word importance."""
        results = []
        
        # initialize word importance calculator
        word_importance = WordImportance(ebook_sentences)
        
        # convert ebook sentences to words with better preprocessing
        ebook_words = []
        sentence_boundaries = []  # keep track of sentence boundaries
        current_pos = 0
        
        for sentence in ebook_sentences:
            # expand contractions and normalize text
            sentence = sentence.lower()
            words = word_importance._clean_text(sentence).split()
            ebook_words.extend(words)
            # store sentence boundary positions
            sentence_boundaries.append((current_pos, current_pos + len(words)))
            current_pos += len(words)
        
        # convert audio words to just text with same preprocessing
        audio_text = []
        for word in audio_words:
            word = word_importance._clean_text(word['text'])
            audio_text.append(word)
        
        logger.info(f"Ebook has {len(ebook_words)} words, audio has {len(audio_text)} words")
        
        # keep track of our position in both texts
        audio_pos = 0
        ebook_pos = 0
        last_match_end = 0
        
        while audio_pos < len(audio_text) - window_size + 1:
            # get current window from audio
            window = audio_text[audio_pos:audio_pos + window_size]
            
            # only search forward from the last match
            search_start = max(last_match_end, ebook_pos - search_window)
            search_end = min(len(ebook_words), ebook_pos + search_window * 2)
            
            # try to find a match with adaptive window size
            best_score = 0
            best_ebook_pos = ebook_pos
            current_window = window_size
            
            while current_window >= 5 and best_score < self.min_confidence:
                window = audio_text[audio_pos:audio_pos + current_window]
                
                for i in range(search_start, search_end - current_window + 1):
                    if i < last_match_end:
                        continue
                        
                    ebook_window = ebook_words[i:i + current_window]
                    
                    # calculate weighted match score
                    weighted_score = word_importance.get_weighted_match_score(window, ebook_window)
                    
                    # apply temporal penalty if match is too far from expected position
                    expected_pos = last_match_end + (audio_pos - len(results))
                    temporal_distance = abs(i - expected_pos)
                    temporal_penalty = min(1.0, temporal_distance / (search_window * 3))
                    adjusted_score = weighted_score * (1 - temporal_penalty * 0.3)  # reduced temporal penalty
                    
                    if adjusted_score > best_score:
                        best_score = adjusted_score
                        best_ebook_pos = i
                
                if best_score >= self.min_confidence:
                    break
                
                # reduce window size more aggressively
                current_window = max(5, int(current_window * 0.8))
            
            if best_score > self.min_confidence:
                logger.info(f"Found match at audio pos {audio_pos} (score: {best_score:.2f})")
                
                # create match result
                start_time = audio_words[audio_pos]['start']
                end_time = audio_words[audio_pos + current_window - 1]['end']
                
                # find the sentence containing these words
                sentence_idx = 0
                for i, (start, end) in enumerate(sentence_boundaries):
                    if best_ebook_pos >= start and best_ebook_pos < end:
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
                
                # update positions with overlap for better continuity
                overlap = min(3, current_window // 2)
                audio_pos += current_window - overlap
                ebook_pos = best_ebook_pos + current_window - overlap
                last_match_end = best_ebook_pos + current_window - overlap
            else:
                # no good match found, advance with smaller step
                audio_pos += 1
        
        return results
    
    def match(self,
              audio_words: List[Dict],
              ebook_sentences: List[str],
              silent_regions: Optional[List[Tuple[float, float]]] = None,
              chapter_num: int = 1) -> List[MatchResult]:
        """
        Match transcribed audio segments with ebook sentences.
        
        Args:
            audio_words: List of word dicts with 'text', 'start', 'end'
            ebook_sentences: List of sentences from ebook
            silent_regions: Optional list of (start, end) silence timestamps
            chapter_num: Chapter number for intro detection
            
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
            if isinstance(sentence, str):  # make sure it's a string
                # look for both chapter and letter markers
                if (f"chapter {chapter_num}" in sentence.lower() or 
                    f"letter {chapter_num}" in sentence.lower() or
                    f"chapter {self._number_to_roman(chapter_num)}" in sentence.lower() or
                    f"letter {self._number_to_roman(chapter_num)}" in sentence.lower()):
                    ebook_start_idx = i + 1  # start after the chapter header
                    logger.info(f"Found chapter/letter start in ebook at index {i}: {sentence}")
                    break
        
        if ebook_start_idx > 0:
            logger.info(f"Skipping {ebook_start_idx} sentences of metadata")
            ebook_sentences = ebook_sentences[ebook_start_idx:]
        
        # detect and skip librivox intro
        chapter_start_idx = self._detect_intro(audio_words, ebook_sentences, chapter_num)
        if chapter_start_idx > 0:
            logger.info(f"Skipping {chapter_start_idx} words of librivox intro")
            audio_words = audio_words[chapter_start_idx:]
        else:
            # fallback: sync start by finding first overlapping word between audio and ebook
            ebook_vocab = set()
            for sent in ebook_sentences:
                for w in sent.split():
                    # only add words that are more than 2 chars to avoid matching numbers/short words
                    if len(w) > 2:
                        ebook_vocab.add(self._preprocess_text(w))
            first_idx = None
            for idx, word in enumerate(audio_words):
                w_clean = self._preprocess_text(word['text'])
                # only match words that are more than 2 chars
                if len(w_clean) > 2 and w_clean in ebook_vocab:
                    first_idx = idx
                    logger.info(f"Sync start: found ebook word '{word['text']}' in audio at index {idx}")
                    break
            if first_idx and first_idx > 0:
                logger.info(f"Skipping intro audio until first ebook word at index {first_idx}")
                audio_words = audio_words[first_idx:]
        
        # try word-level convolution matching with a smaller search window
        results = self._word_convolution_match(audio_words, ebook_sentences, search_window=50)
        
        if not results:
            logger.warning("No matches found with word convolution")
        
        return results

    def enhanced_word_convolution_match(self, audio_words, ebook_words):
        # find anchor words
        anchor_words = find_anchor_words(ebook_words, audio_words)
        
        # use anchors to guide initial matching
        anchor_matches = anchor_based_matching(audio_words, ebook_words, anchor_words)
        
        # sort matches by position
        anchor_matches.sort(key=lambda x: x['audio_pos'])
        
        # fill in gaps between anchor matches
        results = []
        for i in range(len(anchor_matches) - 1):
            current = anchor_matches[i]
            next_match = anchor_matches[i + 1]
            
            # get the text between anchors
            audio_segment = audio_words[current['audio_pos']:next_match['audio_pos']]
            ebook_segment = ebook_words[current['ebook_pos']:next_match['ebook_pos']]
            
            # use our existing convolution matching for the segment
            segment_matches = self._word_convolution_match(
                audio_segment, 
                ebook_segment,
                window_size=min(len(audio_segment), len(ebook_segment))
            )
            
            results.extend(segment_matches)
        
        return results 