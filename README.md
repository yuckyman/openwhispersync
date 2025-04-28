# OpenWhisperSync

Lightweight audiobook alignment tool using OpenAI's Whisper.

## Installation

```bash
pip install openwhispersync
```

## Usage

Align an audiobook with its ebook:

```bash
openwhispersync align --audio book.m4b --ebook book.epub --out sync.json
```

Process all chapters in a directory:

```bash
openwhispersync transcribe --audio-dir /path/to/chapters --out transcriptions.json
```

## Features

- Uses OpenAI's Whisper for accurate transcription
- Supports both MP3 and M4B audio formats
- Handles chapter-based audiobooks
- Preserves word-level timestamps
- Efficient fuzzy matching for alignment

## How it Works

1. **Transcribe**: Uses Whisper (tiny model) to transcribe the audio
2. **Split**: Splits the ebook into sentences
3. **Match**: Uses fuzzy matching to align audio segments with text
4. **Output**: Saves alignment data as JSON

## Requirements

- Python 3.9+
- ffmpeg
- whisper (tiny model)
- ebooklib
- rapidfuzz

## License

MIT