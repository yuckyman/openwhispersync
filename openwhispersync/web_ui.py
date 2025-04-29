from flask import Flask, render_template, jsonify, send_from_directory
import json
import os
import logging
# Import the epub parser - use absolute import when running as module
from openwhispersync.ebook import parse_epub

# Point to templates/static folders inside the openwhispersync package directory
app = Flask(__name__, template_folder='templates', static_folder='static')

# Define base directories relative to the workspace root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # Changed to use current file's directory
# Source files (EPUB, Audio, Alignment) expected here:
FILES_DIR = os.path.join(PROJECT_ROOT, 'files')

# --- Configuration --- #
# Name used for EPUB/Audio directory in 'files'
BOOK_FILES_NAME = 'murderbot'
# Name used in alignment filenames
BOOK_ALIGNMENT_NAME = 'all_systems_red'
# Name displayed in the UI 
BOOK_DISPLAY_NAME = 'All Systems Red'

@app.route('/')
def index():
    """Serves the main read-along page."""
    # Pass the display name to the template
    return render_template('read_along.html', book_name=BOOK_DISPLAY_NAME, 
                           files_name=BOOK_FILES_NAME, 
                           alignment_name=BOOK_ALIGNMENT_NAME)

# Helper function to get book-specific paths
def get_book_paths(files_book_name, alignment_book_name):
    # Source files directory uses the 'files_book_name'
    book_files_dir = os.path.join(FILES_DIR, files_book_name)
    ebook_path = os.path.join(book_files_dir, "all_systems_red.epub")  # Use actual epub filename
    # Alignment files are in the book's directory
    return book_files_dir, ebook_path, book_files_dir

# Route now needs to accept the different names
@app.route('/data/<files_name>/<alignment_name>/<int:chapter_num>')
def get_data(files_name, alignment_name, chapter_num):
    """Endpoint to fetch alignment data and actual ebook chapter text."""
    book_files_dir, ebook_path, alignment_base_dir = get_book_paths(files_name, alignment_name)
    # Construct filename using chapter number only
    alignment_filename = f'chapter_{chapter_num}_alignment.json'
    # Look for alignment file in the alignments subdirectory
    alignment_path = os.path.join(alignment_base_dir, 'alignments', alignment_filename)

    try:
        app.logger.info(f"Attempting to load alignment: {alignment_path}")
        with open(alignment_path, 'r') as f:
            alignment_data = json.load(f)

        app.logger.info(f"Attempting to load EPUB: {ebook_path}")
        if not os.path.exists(ebook_path):
             # Try alternate epub name based on alignment name?
             ebook_alt_path = os.path.join(os.path.dirname(ebook_path), f"{alignment_name}.epub")
             app.logger.warning(f"EPUB not found at {ebook_path}. Trying {ebook_alt_path}")
             if not os.path.exists(ebook_alt_path):
                 return jsonify({"error": f"EPUB file not found at {ebook_path} or {ebook_alt_path}"}), 404
             ebook_path = ebook_alt_path # Use alternate path if found
             
        all_sentences, chapter_markers = parse_epub(ebook_path)
        
        # Find start and end sentence index for the requested chapter
        start_sentence_idx = -1
        end_sentence_idx = len(all_sentences)
        
        marker_keys = sorted(chapter_markers.keys())
        found_marker = False
        for i, marker_idx in enumerate(marker_keys):
            # Handle tuple markers - first element is the text
            marker_value = chapter_markers[marker_idx]
            # Get the raw text, convert to lower, strip whitespace
            marker_text_raw = marker_value[0] if isinstance(marker_value, tuple) else str(marker_value)
            marker_text_clean = marker_text_raw.lower().strip()
            
            # More flexible chapter matching
            chapter_num_str = str(chapter_num)
            # Define potential prefixes
            potential_prefixes = [
                f"chapter {chapter_num_str}", 
                chapter_num_str,
                f"letter {chapter_num_str}"
            ]
            num_to_word = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
            if chapter_num in num_to_word:
                potential_prefixes.append(f"chapter {num_to_word[chapter_num]}")
            if chapter_num == 1:
                potential_prefixes.append("begin reading")

            # Check if the cleaned marker text STARTS WITH any potential prefix
            match_found = False
            for prefix in potential_prefixes:
                if marker_text_clean.startswith(prefix):
                    match_found = True
                    break
            
            if match_found:
                start_sentence_idx = marker_idx
                # Determine end index: use next marker's start or end of book
                if i + 1 < len(marker_keys):
                    end_sentence_idx = marker_keys[i+1]
                else:
                    end_sentence_idx = len(all_sentences) # Use total sentences if it's the last chapter
                found_marker = True
                # Log the raw marker text that matched
                app.logger.info(f"Found marker for Chapter {chapter_num}: '{marker_text_raw}' (matched prefix '{prefix}') at sentence index {start_sentence_idx}. End index: {end_sentence_idx}")
                break
        
        if not found_marker:
             # Handle case where chapter 1 has no explicit marker but content starts at 0
             # Check if any marker starts with potential prefixes for chapter 1
             ch1_prefix_found = any(
                 (mv[0] if isinstance(mv, tuple) else str(mv)).lower().strip().startswith(p) 
                 for mv in chapter_markers.values() 
                 for p in ["chapter 1", "1", "letter 1", "chapter one", "begin reading"]
             )
             if chapter_num == 1 and 0 in chapter_markers and not ch1_prefix_found:
                 start_sentence_idx = 0
                 if marker_keys:
                     end_sentence_idx = marker_keys[0]
                 else:
                     end_sentence_idx = len(all_sentences)
                 found_marker = True # Treat as found
                 app.logger.info(f"Using implicit start for Chapter 1 at sentence index 0.")
             # If still not found, log error
             if not found_marker: 
                 # Log the cleaned markers found for easier debugging
                 cleaned_markers = {k: (v[0] if isinstance(v, tuple) else str(v)).lower().strip() for k, v in chapter_markers.items()}
                 app.logger.warning(f"Chapter marker for Ch {chapter_num} not found. Prefixes checked: {potential_prefixes}. Cleaned markers found: {cleaned_markers}")
                 return jsonify({"error": f"Chapter {chapter_num} marker not found in EPUB TOC/structure."}), 404

        # Ensure start_sentence_idx is not -1 if found_marker is True
        if found_marker and start_sentence_idx == -1:
            app.logger.error(f"Logic error: Found marker for chapter {chapter_num} but start_sentence_idx is -1.")
            return jsonify({"error": "Internal server error finding chapter start."}) , 500

        chapter_sentences = all_sentences[start_sentence_idx:end_sentence_idx]
        adjusted_alignment = []
        for item in alignment_data:
            original_idx = item.get('sentence_idx', -1)
            if start_sentence_idx <= original_idx < end_sentence_idx:
                 new_item = item.copy()
                 new_item['sentence_idx'] = original_idx - start_sentence_idx 
                 adjusted_alignment.append(new_item)

        app.logger.info(f"Returning {len(chapter_sentences)} sentences and {len(adjusted_alignment)} alignment items for Ch {chapter_num}.")
        return jsonify({
            'alignment': adjusted_alignment,
            'sentences': chapter_sentences
        })

    except FileNotFoundError:
        app.logger.error(f"File not found error for Ch {chapter_num}. Tried alignment path: {alignment_path}", exc_info=False)
        return jsonify({"error": f"File not found. Check path: {alignment_path}"}), 404
    except Exception as e:
        app.logger.error(f"Error processing data for {files_name}/{alignment_name} Ch {chapter_num}: {e}", exc_info=True)
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

# Route now needs to accept the files_name for audio path
@app.route('/audio/<files_name>/<int:chapter_num>')
def get_audio(files_name, chapter_num):
    """Serves the audio file for the given book and chapter."""
    book_files_dir, _, _ = get_book_paths(files_name, None)
    
    # Match the actual audio filename pattern
    audio_filename = f"All Systems Red_Part {chapter_num:02d}.mp3"
    audio_path = os.path.join(book_files_dir, audio_filename)
    
    app.logger.info(f"Attempting to serve audio from: {audio_path}")
    
    if not os.path.exists(audio_path):
        app.logger.error(f"Audio file not found for {files_name} Chapter {chapter_num} at {audio_path}")
        return jsonify({"error": f"Audio file for Chapter {chapter_num} not found."}), 404

    try:
        return send_from_directory(directory=book_files_dir, path=os.path.basename(audio_path), as_attachment=False)
    except Exception as e:
        app.logger.error(f"Error serving audio file {audio_path}: {e}", exc_info=True)
        return jsonify({"error": "Failed to serve audio file."}), 500

def main():
    # Enable Flask logging
    logging.basicConfig(level=logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)
    
    app.logger.info("Starting Flask server...")
    app.run(debug=True, port=5001)

if __name__ == '__main__':
    main() 