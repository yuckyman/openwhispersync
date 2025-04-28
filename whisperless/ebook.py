from ebooklib import epub
from lxml import etree
import re
from typing import List
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def parse_epub(epub_path: str) -> List[str]:
    """Extract ordered, cleaned text lines (or sentences) from an EPUB."""
    logger.debug(f"Reading epub from {epub_path}")
    book = epub.read_epub(epub_path)
    
    # Log all items in the book
    logger.debug("All items in the book:")
    for item in book.get_items():
        logger.debug(f"Item ID: {item.get_id()}, Type: {type(item).__name__}, Media Type: {item.media_type}")
    
    spine_ids = [item_id for (item_id, _) in book.spine]                         # preserve order
    logger.debug(f"Found {len(spine_ids)} spine items")
    
    # Get all HTML items, not just EpubHtml
    docs = {item.get_id(): item for item in book.get_items() if item.media_type == 'application/xhtml+xml'}
    logger.debug(f"Found {len(docs)} HTML documents")
    
    text_chunks = []
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
        # extract all paragraph text
        paragraphs = root.xpath('//p')
        logger.debug(f"Found {len(paragraphs)} paragraphs in item {item_id}")
        
        for p in paragraphs:
            # log the raw paragraph content
            logger.debug(f"Raw paragraph content for {item_id}: {etree.tostring(p, encoding='unicode')}")
            txt = ''.join(p.itertext()).strip()
            if txt:
                text_chunks.append(txt)
                logger.debug(f"Extracted text: {txt}")
            else:
                logger.debug(f"No text extracted from paragraph in {item_id}")
    
    logger.debug(f"Extracted {len(text_chunks)} text chunks")
    
    # simple sentence split (fallback regex)
    splitter = re.compile(r'(?<=[.!?])\s+')
    sentences = []
    for chunk in text_chunks:
        for sent in splitter.split(chunk):
            s = sent.strip()
            if s:
                sentences.append(s)
    
    logger.debug(f"Split into {len(sentences)} sentences")
    return sentences

# example usage
if __name__ == '__main__':
    sents = parse_epub('book.epub')
    print(f'Extracted {len(sents)} sentences.')