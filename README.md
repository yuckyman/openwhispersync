# OpenWhisperSync

Lightweight audiobook alignment tool using OpenAI's Whisper.

## Installation

```bash
pip install openwhispersync
```

## Usage

### 1. Transcribe Audio
First, transcribe your audiobook chapters:

```bash
openwhispersync transcribe --audio-dir /path/to/chapters --out transcriptions/
```

This creates a directory with chapter-based transcription files:
```
transcriptions/
├── chapter_1_transcription.json
├── chapter_2_transcription.json
└── ...
```

### 2. Align with Ebook
Next, align the transcriptions with your ebook:

```bash
openwhispersync align --transcriptions transcriptions/ --ebook book.epub --out-dir alignments/
```

This creates a directory with chapter-based alignment files:
```
alignments/
├── chapter_1_alignment.json
├── chapter_2_alignment.json
└── ...
```

### 3. Visualize Results
Finally, generate visualizations to verify the alignment:

```bash
openwhispersync visualize \
  --audio chapter.mp3 \
  --alignment alignments/chapter_1_alignment.json \
  --out-dir visualizations \
  --show
```

## Features

- **Accurate Transcription**: Uses OpenAI's Whisper (base model) for high-quality transcription
- **Smart Alignment**: Hybrid approach combining:
  - Silence detection for natural speech boundaries
  - Fuzzy text matching with rapidfuzz
  - Confidence scoring for alignment quality
- **Format Support**:
  - MP3 and M4B audio formats
  - EPUB ebook format
  - Chapter-based audiobook handling
- **Visualization Tools**:
  - Audio waveform with aligned sentences
  - Silence region highlighting
  - Confidence score visualization
- **Performance Optimized**:
  - Parallel processing for long audio files
  - Memory-efficient processing (<2GB)
  - Fast processing speed (~1.67x realtime)

## How it Works

1. **Audio Processing**:
   - Detect silent regions for natural speech boundaries
   - Transcribe audio using Whisper (base model)
   - Extract word-level timestamps

2. **Text Processing**:
   - Parse EPUB files with proper chapter handling
   - Split text into sentences with smart punctuation handling
   - Preserve chapter markers and structure

3. **Alignment**:
   - Use silence detection to identify potential sentence boundaries
   - Apply fuzzy matching to align audio segments with text
   - Score alignments based on multiple confidence signals

4. **Output**:
   - Save alignment data as JSON
   - Generate visualizations for debugging
   - Support chapter-based organization

## Requirements

- Python 3.9+
- ffmpeg
- whisper (base model)
- ebooklib
- rapidfuzz
- sounddevice (for GUI)
- soundfile (for GUI)
- matplotlib (for visualization)

## License

MIT