from ebooklib import epub
from lxml import etree
import re
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def parse_epub(epub_path: str) -> Tuple[List[str], Dict[int, str]]:
    """
    Extract ordered, cleaned text lines (or sentences) from an EPUB.
    Returns a tuple of (sentences, chapter_markers) where chapter_markers maps
    sentence indices to chapter/letter identifiers.
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
    chapter_markers = {}  # maps sentence index to chapter/letter identifier
    current_sentence_idx = 0
    
    # regex pattern for project gutenberg metadata
    pg_metadata_pattern = re.compile(
        r'Release date: \w+ \d+, \d{4} \[.*?\] Most recently updated: \w+ \d+, \d{4}',
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
            
        # parse XHTML via lxml for speed
        parser = etree.HTMLParser()
        root = etree.fromstring(item.content, parser)
        
        # Skip Project Gutenberg boilerplate
        if item_id in ('pg-header', 'pg-footer', 'coverpage-wrapper'):
            logger.debug(f"Skipping boilerplate item: {item_id}")
            continue
            
        # Extract chapter/letter markers
        chapter_title = root.xpath('//h2/text()')
        if chapter_title:
            chapter_id = chapter_title[0].strip()
            logger.debug(f"Found chapter/letter marker: {chapter_id}")
            chapter_markers[current_sentence_idx] = chapter_id
            
        # extract all paragraph text
        paragraphs = root.xpath('//p')
        logger.debug(f"Found {len(paragraphs)} paragraphs in item {item_id}")
        
        for p in paragraphs:
            # log the raw paragraph content
            logger.debug(f"Raw paragraph content for {item_id}: {etree.tostring(p, encoding='unicode')}")
            txt = ''.join(p.itertext()).strip()
            
            # Skip title headers and metadata, but keep letter and chapter markers
            if txt.lower().startswith(('title:', 'author:', 'chapter', 'letter')):
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
    splitter = re.compile(r'''
        (?<=[.!?])      # lookbehind for sentence enders
        (?!['")\]]\s)   # negative lookahead to avoid splitting inside quotes/parens
        \s+             # one or more whitespace
        (?=[A-Z])       # lookahead for capital letter
    ''', re.VERBOSE)
    
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
    for idx, marker in markers.items():
        print(f'  Sentence {idx}: {marker}')