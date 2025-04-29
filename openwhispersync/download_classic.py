#!/usr/bin/env python3

import os
import sys
import time
from typing import Dict, List, Tuple
import requests
from tqdm import tqdm
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

console = Console()

class ClassicBook:
    def __init__(self, title: str, author: str, ebook_url: str, audiobook_url: str):
        self.title = title
        self.author = author
        self.ebook_url = ebook_url
        self.audiobook_url = audiobook_url

    def __str__(self) -> str:
        return f"{self.title} by {self.author}"

classic_books = [
    ClassicBook(
        "Frankenstein",
        "Mary Shelley",
        "https://www.gutenberg.org/ebooks/84.epub.images",
        "https://www.gutenberg.org/ebooks/84.epub.images"
    ),
    ClassicBook(
        "Walden",
        "Henry David Thoreau",
        "https://www.gutenberg.org/ebooks/205.epub.images",
        "https://www.gutenberg.org/ebooks/205.epub.images"
    ),
    ClassicBook(
        "Self-Reliance",
        "Ralph Waldo Emerson",
        "https://www.gutenberg.org/ebooks/2944.epub.images",
        "https://www.gutenberg.org/ebooks/2944.epub.images"
    ),
    ClassicBook(
        "Pride and Prejudice",
        "Jane Austen",
        "https://www.gutenberg.org/ebooks/1342.epub.images",
        "https://www.gutenberg.org/ebooks/1342.epub.images"
    ),
    ClassicBook(
        "Moby Dick",
        "Herman Melville",
        "https://www.gutenberg.org/ebooks/2701.epub.images",
        "https://www.gutenberg.org/ebooks/2701.epub.images"
    )
]

def download_file(url: str, filename: str) -> None:
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def show_menu() -> None:
    table = Table(title="Available Classic Books")
    table.add_column("Number", style="cyan")
    table.add_column("Title", style="magenta")
    table.add_column("Author", style="green")
    
    for i, book in enumerate(classic_books, 1):
        table.add_row(str(i), book.title, book.author)
    
    console.print(table)

def main():
    console.print("[bold magenta]Welcome to Classic Book Downloader![/bold magenta]")
    console.print("Choose a book to download both the ebook and audiobook versions.")
    
    while True:
        show_menu()
        choice = Prompt.ask("Enter the number of the book you want to download (or 'q' to quit)")
        
        if choice.lower() == 'q':
            console.print("[yellow]Goodbye![/yellow]")
            sys.exit(0)
            
        try:
            choice = int(choice)
            if 1 <= choice <= len(classic_books):
                book = classic_books[choice - 1]
                break
            else:
                console.print("[red]Please enter a valid number from the list.[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number.[/red]")
    
    # Create downloads directory if it doesn't exist
    book_folder = f"openwhispersync/files/{book.title}"
    os.makedirs(book_folder, exist_ok=True)
    
    # Download ebook
    console.print(f"\n[bold]Downloading ebook: {book.title}[/bold]")
    ebook_filename = f"{book_folder}/{book.title.lower().replace(' ', '_')}.epub"
    download_file(book.ebook_url, ebook_filename)
    
    # Download audiobook
    console.print(f"\n[bold]Downloading audiobook: {book.title}[/bold]")
    audiobook_filename = f"{book_folder}/{book.title.lower().replace(' ', '_')}.mp3"
    download_file(book.audiobook_url, audiobook_filename)
    
    console.print(f"\n[green]Successfully downloaded {book.title}![/green]")
    console.print(f"Ebook saved as: {ebook_filename}")
    console.print(f"Audiobook saved as: {audiobook_filename}")

if __name__ == "__main__":
    main() 