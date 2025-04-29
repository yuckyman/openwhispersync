from ebooklib import epub
from lxml import etree
import re
from typing import List, Dict, Tuple, Optional
import logging
from bs4 import BeautifulSoup
import json

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def normalize_text(text):
    """Normalize text by standardizing quotes, dashes, and whitespace."""
    # normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # fix common number formatting issues
    text = re.sub(r'(\d+)\s*,\s*(\d+)', r'\1,\2', text)
    # normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    # normalize dashes
    text = text.replace('–', '-').replace('—', '-')
    # normalize ellipsis
    text = text.replace('…', '...')
    return text.strip()

def should_skip_item(item):
    """Check if an epub item should be skipped."""
    if not isinstance(item, epub.EpubHtml):
        return True
        
    # skip common metadata sections
    skip_patterns = [
        r'copyright',
        r'title\s*page',
        r'contents',
        r'dedication',
        r'acknowledgments?',
        r'about\s*the\s*author',
    ]
    
    soup = BeautifulSoup(item.content, 'html.parser')
    text = soup.get_text().lower()
    
    return any(re.search(pattern, text) for pattern in skip_patterns)

def parse_chapter_number(marker: str) -> Optional[int]:
    """Extract chapter number from various marker formats."""
    # lowercase and clean
    text = marker.lower().strip()
    
    # common patterns to try
    patterns = [
        # arabic numbers: "chapter 1", "letter 2"
        r'(?:chapter|letter|part|book)\s+(\d+)',
        # roman numerals: "chapter iv"
        r'(?:chapter|letter|part|book)\s+((?:x{0,3})(?:ix|iv|v?i{0,3}))$',
        # spelled out: "chapter one"
        r'(?:chapter|letter|part|book)\s+(one|two|three|four|five|six|seven|eight|nine|ten)',
        # just the number: "1.", "1:", "1"
        r'^(\d+)[.:]?\s*$'
    ]
    
    # number word mapping
    num_words = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    
    # roman numeral mapping
    roman_map = {
        'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5,
        'vi': 6, 'vii': 7, 'viii': 8, 'ix': 9, 'x': 10
    }
    
    for pattern in patterns:
        if match := re.search(pattern, text):
            num = match.group(1).lower()
            # convert word to number
            if num in num_words:
                return num_words[num]
            # convert roman to number
            if num in roman_map:
                return roman_map[num]
            # try direct integer conversion
            try:
                return int(num)
            except ValueError:
                continue
    
    return None

def parse_epub(epub_path: str) -> Tuple[List[str], Dict[int, Tuple[str, int]]]:
    """
    Extract ordered, cleaned text lines (or sentences) from an EPUB.
    Returns a tuple of (sentences, chapter_markers) where chapter_markers maps
    sentence indices to (marker_text, chapter_number) identifiers.
    """
    logger.debug(f"Reading epub from {epub_path}")
    book = epub.read_epub(epub_path)
    
    # Log all items in the book
    logger.debug("All items in the book:")
    for item in book.get_items():
        logger.debug(f"Item ID: {item.get_id()}, Type: {type(item).__name__}, Media Type: {item.media_type}")
    
    spine_ids = [item_id for (item_id, _) in book.spine]  # preserve order
    logger.debug(f"Found {len(spine_ids)} spine items")
    
    # Get all HTML items, not just EpubHtml
    docs = {item.get_id(): item for item in book.get_items() if item.media_type == 'application/xhtml+xml'}
    logger.debug(f"Found {len(docs)} HTML documents")
    
    text_chunks = []
    chapter_markers = {}  # maps sentence index to (marker_text, chapter_number)
    current_sentence_idx = 0
    
    # regex pattern for project gutenberg metadata
    pg_metadata_pattern = re.compile(
        r'Release date: \w+ \d+, \d{4} \[.*?\] Most recently updated: \w+ \d+, \d{4}',
        re.IGNORECASE
    )
    
    # regex pattern for chapter markers
    chapter_pattern = re.compile(r'^(?:chapter|letter)\s+\w+', re.IGNORECASE)
    
    # sort spine items to ensure chapters are in order
    spine_ids.sort(key=lambda x: int(re.search(r'ch(\d+)', x).group(1)) if re.search(r'ch(\d+)', x) else float('inf'))
    
    for item_id in spine_ids:
        item = docs.get(item_id)
        if not item:
            logger.debug(f"No item found for spine ID {item_id}")
            continue
        if not item.content:
            logger.debug(f"No content found for item {item_id}")
            continue
            
        # parse XHTML via lxml for speed
        parser = etree.HTMLParser()
        root = etree.fromstring(item.content, parser)
        
        # Skip Project Gutenberg boilerplate
        if item_id in ('pg-header', 'pg-footer', 'coverpage-wrapper'):
            logger.debug(f"Skipping boilerplate item: {item_id}")
            continue
            
        # Extract chapter/letter markers - try multiple strategies
        # 1. Look for div with class 'chapter' and specific id pattern
        chapter_elements = root.xpath('//div[@class="chapter" and starts-with(@id, "ch")]')
        if chapter_elements:
            chapter_id = chapter_elements[0].get('id')
            # get the heading text from within this div
            chapter_title = chapter_elements[0].xpath('.//h1[@class="chapter"]//text()')
            if chapter_title:
                chapter_id = chapter_title[0].strip()
            logger.debug(f"Found chapter marker from div.chapter: {chapter_id}")
            chapter_markers[current_sentence_idx] = (chapter_id, parse_chapter_number(chapter_id))
        
        # if no chapter found, try previous methods
        if not chapter_elements:
            # 2. Look for h1/h2 with class 'chapter'
            chapter_title = root.xpath('//h1[@class="chapter"]/text() | //h2[@class="chapter"]/text()')
            if not chapter_title:
                # 3. Look for any h1/h2 containing the word 'chapter'
                chapter_title = root.xpath('//h1[contains(translate(text(), "CHAPTER", "chapter"), "chapter")]/text() | //h2[contains(translate(text(), "CHAPTER", "chapter"), "chapter")]/text()')
            if not chapter_title:
                # 4. Look for chapter in the filename
                chapter_match = re.search(r'ch(\d+)', item_id)
                if chapter_match:
                    chapter_title = [f"Chapter {chapter_match.group(1)}"]
                
            if chapter_title:
                chapter_id = chapter_title[0].strip()
                logger.debug(f"Found chapter/letter marker: {chapter_id}")
                chapter_markers[current_sentence_idx] = (chapter_id, parse_chapter_number(chapter_id))
            
        # extract all paragraph text
        paragraphs = root.xpath('//p')
        logger.debug(f"Found {len(paragraphs)} paragraphs in item {item_id}")
        
        for p in paragraphs:
            # log the raw paragraph content
            logger.debug(f"Raw paragraph content for {item_id}: {etree.tostring(p, encoding='unicode')}")
            txt = ''.join(p.itertext()).strip()
            
            # Skip title headers and metadata, but keep letter and chapter markers
            if txt.lower().startswith(('title:', 'author:')):
                logger.debug(f"Skipping header/metadata: {txt}")
                continue
                
            # Skip project gutenberg metadata lines
            if pg_metadata_pattern.match(txt):
                logger.debug(f"Skipping project gutenberg metadata: {txt}")
                continue
                
            if txt:
                text_chunks.append(txt)
                logger.debug(f"Extracted text: {txt}")
            else:
                logger.debug(f"No text extracted from paragraph in {item_id}")
    
    logger.debug(f"Extracted {len(text_chunks)} text chunks")
    
    # improved sentence splitting regex
    # handles:
    # - embedded newlines
    # - unusual punctuation
    # - multiple spaces
    # - quotes and parentheses
    splitter = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    
    sentences = []
    for chunk in text_chunks:
        # normalize whitespace first
        chunk = ' '.join(chunk.split())
        for sent in splitter.split(chunk):
            s = sent.strip()
            if s:
                sentences.append(s)
                current_sentence_idx += 1
    
    logger.debug(f"Split into {len(sentences)} sentences")
    logger.debug(f"Found {len(chapter_markers)} chapter/letter markers")
    return sentences, chapter_markers

# example usage
if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m openwhispersync.ebook <epub_file>")
        sys.exit(1)
    epub_path = sys.argv[1]
    sents, markers = parse_epub(epub_path)
    print(f'Extracted {len(sents)} sentences.')
    print(f'Found {len(markers)} chapter/letter markers:')
    for idx, (marker_text, chapter_num) in markers.items():
        print(f'  Sentence {idx}: {marker_text} (Chapter {chapter_num})')