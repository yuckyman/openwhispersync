import re
import math
import logging
from typing import List, TypedDict, Optional, Tuple
from dataclasses import dataclass
from rapidfuzz import fuzz, process
from rich.progress import Progress

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
    idx: int
    start_time: float
    end_time: float
    confidence: float
    matched_text: str
    is_silence_based: bool = False

def clean_for_matching(text: str) -> str:
    """Prepare text for fuzzy matching by lowercasing and removing most punctuation (except apostrophes)."""
    txt = text.lower()
    txt = re.sub(r"[^a-z0-9'\s]+", ' ', txt)
    return re.sub(r'\s+', ' ', txt).strip()

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
    def __init__(self,
                 audio: List[AudioWord],
                 ebook: List[str],
                 min_conf: float = 0.7):
        self.audio = audio
        self.raw_sent = ebook
        self.proc_sent = [clean_for_matching(s) for s in ebook]
        self.min_conf = min_conf
        self.imp = WordImportance(ebook)
        self.silent_regions = []

    def _audio_window(self, pos: int, size: int = 10):
        chunk = self.audio[pos:pos+size]
        text = ' '.join(clean_for_matching(w['text']) for w in chunk)
        return text, chunk[0]['start'], chunk[-1]['end']

    def match(
        self,
        audio_words: List[AudioWord],
        ebook_sentences: List[str],
        silent_regions: Optional[List[Tuple[float, float]]] = None,
        chapter_num: int = 1
    ) -> List[MatchResult]:
        # reassign to instance so existing logic works
        self.audio = audio_words
        self.raw_sent = ebook_sentences
        self.proc_sent = [clean_for_matching(s) for s in ebook_sentences]
        self.min_conf = self.min_conf  # unchanged
        self.imp = WordImportance(ebook_sentences)
        self.silent_regions = silent_regions or []

        # now call the old matching logic
        return self._run_matching()

    def _run_matching(self) -> List[MatchResult]:
        results = []
        
        # first try silence-based matching
        if self.silent_regions:
            for start, end in self.silent_regions:
                # find words that occur right before silence
                pre_silence_words = [w for w in self.audio 
                                   if w['end'] <= start and w['end'] > start - 2.0]
                
                # find words right after silence
                post_silence_words = [w for w in self.audio 
                                    if w['start'] >= end and w['start'] < end + 2.0]
                
                if pre_silence_words and post_silence_words:
                    # create text windows around silence
                    pre_text = ' '.join(w['text'] for w in pre_silence_words[-5:])
                    post_text = ' '.join(w['text'] for w in post_silence_words[:5])
                    
                    # try to match these against sentence boundaries in the text
                    for i, (sent1, sent2) in enumerate(zip(self.proc_sent[:-1], self.proc_sent[1:])):
                        # check if pre-silence matches end of sent1
                        # and post-silence matches start of sent2
                        if (fuzz.partial_ratio(clean_for_matching(pre_text), sent1[-50:]) > 80 and
                            fuzz.partial_ratio(clean_for_matching(post_text), sent2[:50]) > 80):
                            
                            results.append(MatchResult(
                                sentence=self.raw_sent[i],
                                idx=i,
                                start_time=pre_silence_words[0]['start'],
                                end_time=post_silence_words[-1]['end'],
                                confidence=0.9,  # high confidence for silence-based
                                matched_text=pre_text + " ... " + post_text,
                                is_silence_based=True
                            ))
        
        # then do regular window-based matching
        for i in Progress().track(range(len(self.audio)), description="Matching"):
            window, t0, t1 = self._audio_window(i)
            best = process.extractOne(
                window,
                self.proc_sent,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=int(self.min_conf * 100)
            )
            if not best:
                continue
            match_txt, score, idx = best
            common = set(window.split()) & set(self.proc_sent[idx].split())
            imp_factor = sum(self.imp.score(w) for w in common) / max(1, len(common))
            confidence = min(1.0, (score / 100) * imp_factor)
            if confidence >= self.min_conf:
                results.append(MatchResult(
                    sentence=self.raw_sent[idx],
                    idx=idx,
                    start_time=t0,
                    end_time=t1,
                    confidence=confidence,
                    matched_text=window,
                    is_silence_based=False  # default to False for now
                ))
        return results