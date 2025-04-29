from ebooklib import epub
import ebooklib
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
    
    # --- Enhanced TOC Parsing --- 
    toc_items = []
    # Try common TOC item IDs first
    toc_id_candidates = ['toc', 'ncx', 'nav'] 
    for item_id in toc_id_candidates:
        item = book.get_item_with_id(item_id)
        if item:
            logger.debug(f"Found potential TOC item by ID: {item_id}")
            try:
                parser = etree.HTMLParser()
                root = etree.fromstring(item.content, parser)
                # Look for any links within common TOC structures
                toc_items = root.xpath(
                    '//nav[@epub:type="toc"]//a | //nav[contains(@id, "toc")]//a | //div[contains(@id, "toc")]//a | //body//a'
                )
                if toc_items:
                    logger.debug(f"Found {len(toc_items)} links in TOC item {item_id}")
                    break # Found links, stop searching TOC items
            except Exception as e:
                logger.warning(f"Error parsing potential TOC item {item_id}: {e}")
    
    # If still no items, try searching all HTML items for TOC patterns
    if not toc_items:
        logger.debug("No TOC links found via common IDs, searching all HTML items...")
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
             try:
                parser = etree.HTMLParser()
                root = etree.fromstring(item.content, parser)
                # Broader search for links that might be chapter links
                potential_links = root.xpath('//a[re:test(@href, "(?:ch|chap|part|sec)\\d*\\.", "i")] | //a[re:test(text(), "(?:chapter|part|section|letter)", "i")]', 
                                             namespaces={'re': "http://exslt.org/regular-expressions"})
                if potential_links:
                    logger.debug(f"Found {len(potential_links)} potential TOC links in item {item.get_id()}")
                    toc_items.extend(potential_links)
             except Exception as e:
                 logger.warning(f"Error parsing item {item.get_id()} for TOC links: {e}")
        # Remove duplicates if searching multiple files found them
        toc_items = list({etree.tostring(item): item for item in toc_items}.values())
        logger.debug(f"Found {len(toc_items)} unique potential TOC links across all items.")

    # Map of chapter file names (without extension) to chapter numbers
    chapter_map = {}
    current_chapter_num = 1
    for toc_link in toc_items:
        href = toc_link.get('href', '')
        link_text = ''.join(toc_link.itertext()).strip()
        
        # Try to parse chapter number from link text first
        parsed_num = parse_chapter_number(link_text)
        
        # Extract base filename from href
        base_filename = href.split('#')[0].split('.')[0] if href else None

        if base_filename:
            if base_filename not in chapter_map:
                # Use parsed number if available, otherwise assign sequentially
                num_to_assign = parsed_num if parsed_num is not None else current_chapter_num
                chapter_map[base_filename] = num_to_assign
                logger.debug(f"Mapped TOC base file '{base_filename}' to Chapter {num_to_assign} (from link text: '{link_text}')")
                # Only increment sequence if we didn't get a number from text
                if parsed_num is None:
                     current_chapter_num += 1
            # Handle cases where the same file might be linked multiple times (e.g. subsections)
            # We prioritize the first assignment based on TOC order.
        elif parsed_num is not None:
            # If no filename but we parsed a number from text, log it but don't map yet
            # This might be useful later if we need more complex logic
            logger.debug(f"Parsed chapter number {parsed_num} from TOC link text '{link_text}' but no file href found.")
            # TODO: Could potentially map this number sequentially if needed?

    spine_ids = [item_id for (item_id, _) in book.spine]  # preserve order
    logger.debug(f"Found {len(spine_ids)} spine items")
    
    # Get all HTML items, not just EpubHtml
    docs = {item.get_id(): item for item in book.get_items() if item.media_type == 'application/xhtml+xml'}
    logger.debug(f"Found {len(docs)} HTML documents")
    
    text_chunks = []
    chapter_markers = {}  # maps sentence index to (marker_text, chapter_number)
    current_sentence_idx = 0
    processed_chapter_nums = set()
    
    # regex pattern for project gutenberg metadata
    pg_metadata_pattern = re.compile(
        r'Release date: \w+ \d+, \d{4} \[.*?\] Most recently updated: \w+ \d+, \d{4}',
        re.IGNORECASE
    )
    chapter_pattern = re.compile(
        r'^(?:chapter|part|section|book|letter)\s+\w+|^\d+\.?\s*$|^[IVXLC]+\.?\s*$',
        re.IGNORECASE
    )

    for item_id in spine_ids:
        item = docs.get(item_id)
        if not item:
            logger.debug(f"No item found for spine ID {item_id}")
            continue
        if not item.content:
            logger.debug(f"No content found for item {item_id}")
            continue
            
        # Skip Project Gutenberg boilerplate or other non-content items
        # Made this check more generic
        item_text_lower = item.get_body_content().decode('utf-8', errors='ignore').lower()
        skip_keywords = ['toc', 'table of contents', 'copyright', 'dedication', 'title page', 'index', 'appendix', 'glossary']
        if any(keyword in item_text_lower for keyword in skip_keywords) and len(item_text_lower) < 1000: # Skip short, likely non-content pages
             logger.debug(f"Skipping potential non-content item: {item_id}")
             continue

        # --- Check if this item corresponds to a chapter from TOC --- 
        item_base_filename = item.file_name.split('.')[0]
        chapter_num_from_toc = chapter_map.get(item_base_filename)
        
        # Mark the beginning of the chapter content using TOC info if available
        # Only mark the *first* time we encounter a spine item for a given chapter number
        if chapter_num_from_toc is not None and chapter_num_from_toc not in processed_chapter_nums:
            marker_text = f"Chapter {chapter_num_from_toc} (from TOC)"
            # Use current_sentence_idx BEFORE processing this item's sentences
            chapter_markers[current_sentence_idx] = (marker_text, chapter_num_from_toc)
            processed_chapter_nums.add(chapter_num_from_toc)
            logger.debug(f"Marked Chapter {chapter_num_from_toc} at sentence index {current_sentence_idx} (based on TOC mapping for file '{item.file_name}')")

        # parse XHTML via lxml for speed
        parser = etree.HTMLParser()
        try:
            root = etree.fromstring(item.content, parser)
        except Exception as e:
            logger.error(f"Failed to parse item {item_id}: {e}")
            continue # Skip item if parsing fails

        paragraphs = root.xpath('//p')
        logger.debug(f"Found {len(paragraphs)} paragraphs in item {item_id}")
        
        item_sentences = [] # Collect sentences for THIS item
        for p in paragraphs:
            # log the raw paragraph content
            # logger.debug(f"Raw paragraph content for {item_id}: {etree.tostring(p, encoding='unicode')}")
            txt = ''.join(p.itertext()).strip()
            
            # Skip headers/metadata more carefully
            if txt.lower().startswith(('title:', 'author:')) or pg_metadata_pattern.match(txt):
                # logger.debug(f"Skipping header/metadata: {txt}")
                continue
                
            if txt:
                item_sentences.extend(split_text_into_sentences(txt))
            # else:
                # logger.debug(f"No text extracted from paragraph in {item_id}")
        
        # Add sentences from this item to the main list and update index
        if item_sentences:
            text_chunks.extend(item_sentences) # Add sentences, not chunks
            current_sentence_idx += len(item_sentences)
            logger.debug(f"Added {len(item_sentences)} sentences from item {item_id}. Total sentences now: {current_sentence_idx}")
        else:
            logger.debug(f"No sentences added from item {item_id}")

    # Use the collected sentences directly (no re-splitting needed)
    sentences = text_chunks 
    logger.debug(f"Final total sentences: {len(sentences)}")

    # --- Fallback Marker Detection (if TOC failed) --- 
    if not chapter_markers: 
        logger.info("TOC-based chapter marking failed or produced no markers. Using fallback text search...")
        # ... (keep the improved fallback logic from the previous step) ...
        found_markers_temp = {}
        for i, sent in enumerate(sentences):
            if chapter_pattern.match(sent.strip()): 
                if num := parse_chapter_number(sent):
                    if num not in found_markers_temp:
                        found_markers_temp[num] = (sent, i) 
                        logger.debug(f"Found potential chapter marker from text: '{sent}' -> {num} at index {i}")
        
        sorted_nums = sorted(found_markers_temp.keys())
        for num in sorted_nums:
            marker_text, sentence_idx = found_markers_temp[num]
            chapter_markers[sentence_idx] = (marker_text, num)
            logger.debug(f"Using chapter marker: Index {sentence_idx} for Chapter {num}")

    logger.info(f"Final chapter markers found: {len(chapter_markers)}")
    return sentences, chapter_markers

def split_text_into_sentences(text: str) -> List[str]:
    """Helper function to split a text chunk into sentences."""
    splitter = re.compile(r'(?<=[.!?])\s+(?=[A-Z\"\'\(\[])') # Slightly improved splitter
    sentences = []
    text = ' '.join(text.split()) # normalize whitespace
    for sent in splitter.split(text):
        s = sent.strip()
        if s:
            sentences.append(s)
    return sentences

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