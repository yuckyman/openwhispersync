 # Whisperless

Lightweight CLI tool for aligning audiobooks with their ebook counterparts.

## Installation

```bash
pip install whisperless
```

## Usage

```bash
whisperless align --audio book.m4b --ebook book.epub --out sync.json
```

### Options

- `--audio`: Path to audio file (required)
- `--ebook`: Path to ebook file (required)
- `--out`: Output JSON file (default: sync.json)
- `--chunk-size`: Processing chunk size (default: 5m)

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