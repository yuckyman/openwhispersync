import sys
from pathlib import Path
from collections import defaultdict
import re

# Add the parent directory to the path so we can import whisperless
sys.path.append(str(Path(__file__).parent.parent))

from whisperless.core import split_ebook
from rich.console import Console
from rich.table import Table

console = Console()

def count_words(text: str) -> int:
    """Count words in text, handling contractions and hyphenated words properly."""
    # Replace hyphens with spaces but keep contractions
    text = re.sub(r'(?<!\')\b-\b(?!\')', ' ', text)
    # Split on whitespace and filter out empty strings
    words = [w for w in text.split() if w]
    return len(words)

def analyze_epub():
    epub_path = Path(__file__).parent.parent / "files" / "frankenstein" / "frankenstein.epub"
    
    console.print("[bold green]Analyzing Frankenstein epub...[/bold green]")
    console.print(f"Processing: {epub_path}")
    
    try:
        # Get sentences from epub
        sentences = split_ebook(str(epub_path))
        
        # Debug: print first few sentences
        print("\nFirst 10 sentences:")
        for i, sentence in enumerate(sentences[:20]):
            print(f"{i}: {sentence}")
        print()
        
        # Analyze by chapter
        chapter_stats = defaultdict(lambda: {"sentences": 0, "words": 0})
        current_chapter = 0
        
        # Roman numeral pattern
        roman_pattern = r'\b(?:Letter|Chapter)\s+(?:I{1,3}|IV|V|VI{1,3}|IX|X|XI{1,3}|XIV|XV|XVI{1,3}|XIX|XX|XXI{1,3}|XXIV)\b'
        
        for sentence in sentences:
            # Look for chapter markers using roman numerals
            if re.search(roman_pattern, sentence, re.IGNORECASE):
                current_chapter += 1
                continue
                
            chapter_stats[current_chapter]["sentences"] += 1
            chapter_stats[current_chapter]["words"] += count_words(sentence)
        
        # Create a pretty table
        table = Table(title="Frankenstein Chapter Statistics")
        table.add_column("Chapter", justify="right", style="cyan")
        table.add_column("Sentences", justify="right", style="magenta")
        table.add_column("Words", justify="right", style="green")
        table.add_column("Avg Words/Sentence", justify="right", style="yellow")
        
        total_sentences = 0
        total_words = 0
        
        for chapter, stats in sorted(chapter_stats.items()):
            if chapter == 0:  # Skip preface/intro
                continue
                
            avg_words = stats["words"] / stats["sentences"] if stats["sentences"] > 0 else 0
            table.add_row(
                str(chapter),
                str(stats["sentences"]),
                str(stats["words"]),
                f"{avg_words:.1f}"
            )
            total_sentences += stats["sentences"]
            total_words += stats["words"]
        
        # Add totals
        avg_words_per_sentence = total_words/total_sentences if total_sentences > 0 else 0
        table.add_row(
            "TOTAL",
            str(total_sentences),
            str(total_words),
            f"{avg_words_per_sentence:.1f}",
            style="bold"
        )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error during analysis: {e}[/red]")
        raise

if __name__ == "__main__":
    analyze_epub() 