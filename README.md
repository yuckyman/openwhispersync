# OpenWhisperSync

Lightweight audiobook alignment tool using OpenAI's Whisper.

openwhispersync is a lightweight tool that magically syncs your audiobooks with their ebook counterparts. powered by openai's whisper, it creates perfect word-by-word alignments that make following along a breeze.

## quick start

```bash
# install
pip install openwhispersync

# transcribe your audiobook chapters
openwhispersync transcribe --audio-dir /path/to/chapters --out transcriptions/

# align with your ebook
openwhispersync align --transcriptions transcriptions/ --ebook book.epub --out-dir alignments/

# launch the web ui and start reading!
python -m openwhispersync.web_ui
```

## features

- **accurate transcription** using openai's whisper (base model)
- **smart alignment** with:
  - silence detection for natural speech boundaries
  - fuzzy text matching with rapidfuzz
  - confidence scoring for alignment quality
- **format support**:
  - mp3 and m4b audio formats
  - epub ebook format
  - chapter-based audiobook handling
- **visualization tools** for debugging and verification
- **performance optimized**:
  - parallel processing for long audio files
  - memory-efficient processing (<2gb)
  - fast processing speed (~1.67x realtime)

## web ui (read-along feature)

the web ui is where the magic happens! it provides a beautiful, synchronized read-along experience.

### setup

1. **prepare your files**:
   - place your epub, chapter audio files (e.g., `Chapter_1.mp3`), and alignment json files in:
     ```
     openwhispersync/files/<book_name>/
     ```
   - (you might need to adjust the default book name in `web_ui.py`)

2. **launch the ui**:
   ```bash
   python -m openwhispersync.web_ui
   ```

3. **open your browser** to [http://127.0.0.1:5001/](http://127.0.0.1:5001/)

### features
- synchronized text highlighting
- audio playback controls
- chapter navigation
- clean, modern interface

## detailed usage

### 1. transcribe audio
```bash
openwhispersync transcribe --audio-dir /path/to/chapters --out transcriptions/
```

creates:
```
transcriptions/
├── chapter_1_transcription.json
├── chapter_2_transcription.json
└── ...
```

### 2. align with ebook
```bash
openwhispersync align --transcriptions transcriptions/ --ebook book.epub --out-dir alignments/
```

creates:
```
alignments/
├── chapter_1_alignment.json
├── chapter_2_alignment.json
└── ...
```

### 3. visualize results
```bash
openwhispersync visualize \
  --audio chapter.mp3 \
  --alignment alignments/chapter_1_alignment.json \
  --out-dir visualizations \
  --show
```

## requirements

- python 3.9+
- ffmpeg
- whisper (base model)
- ebooklib
- rapidfuzz
- sounddevice (for gui)
- soundfile (for gui)
- matplotlib (for visualization)

## troubleshooting

### common issues

1. **audio format issues**
   - ensure your audio files are in mp3 or m4b format
   - use ffmpeg to convert if needed: `ffmpeg -i input.m4a output.mp3`

2. **web ui not starting**
   - check if port 5001 is available
   - verify all required files are in the correct directory structure

3. **alignment quality**
   - try adjusting the silence threshold in the config
   - ensure audio quality is good (no background noise)

## license

mit