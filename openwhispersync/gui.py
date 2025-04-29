import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import json
from pathlib import Path
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
from openwhispersync.ebook import parse_epub
# from matplotlib.figure import Figure # will add later for waveform
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # will add later

class SyncViewerApp:
    def __init__(self, master):
        self.master = master
        master.title("OpenWhisperSync Viewer")
        master.geometry("800x600")

        self.audio_data = None
        self.ebook_sentences = None
        self.ebook_chapter_markers = None  # new: store chapter markers
        self.alignment_data = None
        self.current_chapter = tk.IntVar(value=1) # default to chapter 1
        self.audio_file_path = None
        self.audio_duration = 0
        self.playback_thread = None
        self.is_playing = False
        self.current_time = 0.0
        self.playback_event = threading.Event()  # new: for controlling playback thread
        self.audio_stream = None  # new: for audio playback
        self.audio_data = None  # new: for audio samples

        # --- Top Frame: File Loading ---
        top_frame = ttk.Frame(master, padding="10")
        top_frame.pack(fill=tk.X)

        ttk.Button(top_frame, text="Load Audio JSON", command=self.load_audio_json).pack(side=tk.LEFT, padx=5)
        self.audio_json_label = ttk.Label(top_frame, text="No audio JSON loaded")
        self.audio_json_label.pack(side=tk.LEFT, padx=5)

        ttk.Button(top_frame, text="Load Ebook (.epub)", command=self.load_ebook).pack(side=tk.LEFT, padx=5)
        self.ebook_label = ttk.Label(top_frame, text="No ebook loaded")
        self.ebook_label.pack(side=tk.LEFT, padx=5)

        # --- Middle Frame: Chapter Selection & Controls ---
        middle_frame = ttk.Frame(master, padding="10")
        middle_frame.pack(fill=tk.X)

        ttk.Label(middle_frame, text="Chapter:").pack(side=tk.LEFT, padx=5)
        self.chapter_selector = ttk.Combobox(middle_frame, textvariable=self.current_chapter, state="disabled", width=5)
        self.chapter_selector.pack(side=tk.LEFT, padx=5)
        self.chapter_selector.bind("<<ComboboxSelected>>", self.load_chapter_data) # Load data when chapter changes

        self.play_pause_button = ttk.Button(middle_frame, text="▶ Play", command=self.toggle_playback, state="disabled")
        self.play_pause_button.pack(side=tk.LEFT, padx=5)

        self.time_label = ttk.Label(middle_frame, text="00:00 / 00:00")
        self.time_label.pack(side=tk.LEFT, padx=5)

        self.seek_slider = ttk.Scale(middle_frame, from_=0, to=100, orient=tk.HORIZONTAL, length=400, command=self.seek_audio, state="disabled")
        self.seek_slider.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)


        # --- Bottom Frame: Text Display ---
        bottom_frame = ttk.Frame(master, padding="10")
        bottom_frame.pack(fill=tk.BOTH, expand=True)

        self.text_area = scrolledtext.ScrolledText(bottom_frame, wrap=tk.WORD, state="disabled", font=("Arial", 12))
        self.text_area.pack(fill=tk.BOTH, expand=True)
        # Add tags for highlighting
        self.text_area.tag_configure("highlight", background="yellow")
        self.text_area.tag_configure("dim", foreground="gray") # for text outside current sentence


    def load_audio_json(self):
        filepath = filedialog.askopenfilename(
            title="Select Audio JSON File",
            filetypes=[("JSON files", "*.json")]
        )
        if not filepath:
            return

        try:
            with open(filepath, 'r') as f:
                self.audio_data = json.load(f)
            self.audio_json_label.config(text=Path(filepath).name)
            self._update_chapter_selector()
            # Attempt to load alignment for the default chapter
            self.load_chapter_data()
            print(f"Loaded audio JSON: {filepath}")
        except Exception as e:
            print(f"Error loading audio JSON: {e}")
            self.audio_data = None
            self.audio_json_label.config(text="Error loading file")
            self.chapter_selector.config(state="disabled")

    def load_ebook(self):
        filepath = filedialog.askopenfilename(
            title="Select Ebook File",
            filetypes=[("EPUB files", "*.epub")]
        )
        if not filepath:
            return

        try:
            # Parse the epub file
            self.ebook_path = Path(filepath)
            self.ebook_sentences, self.ebook_chapter_markers = parse_epub(self.ebook_path)
            self.ebook_label.config(text=self.ebook_path.name)
            print(f"Successfully parsed ebook: {len(self.ebook_sentences)} sentences")
            
            # Try to load chapter data if audio is already loaded
            if self.audio_data:
                self.load_chapter_data()
        except Exception as e:
            print(f"Error parsing epub: {e}")
            self.ebook_sentences = None
            self.ebook_chapter_markers = None
            self.ebook_label.config(text="Error loading file")


    def _update_chapter_selector(self):
        if self.audio_data and "chapters" in self.audio_data:
            chapter_numbers = [chap["number"] for chap in self.audio_data["chapters"]]
            self.chapter_selector.config(values=chapter_numbers, state="readonly")
            if chapter_numbers:
                self.current_chapter.set(chapter_numbers[0]) # Set to first chapter
        else:
            self.chapter_selector.config(values=[], state="disabled")
            self.current_chapter.set(0)


    def load_chapter_data(self, event=None):
        if not self.audio_data or not self.ebook_sentences:
            print("Audio or ebook data not loaded")
            self._disable_controls()
            return

        selected_chapter = self.chapter_selector.get()
        if not selected_chapter:
            print("No chapter selected")
            self._disable_controls()
            return

        # Extract chapter number from the selection
        try:
            selected_chapter_num = int(selected_chapter)
            print(f"Selected chapter number: {selected_chapter_num}")
        except ValueError:
            print(f"Invalid chapter number: {selected_chapter}")
            self._disable_controls()
            return

        # Find chapter info in audio data
        chapter_info = next((chap for chap in self.audio_data["chapters"] if chap["number"] == selected_chapter_num), None)

        if not chapter_info:
            print(f"Chapter {selected_chapter_num} not found in audio JSON.")
            self._disable_controls()
            return

        # Get chapter sentences using chapter markers
        # Find the start of this chapter
        chapter_start = 0
        for idx, marker in self.ebook_chapter_markers.items():
            if f"chapter {selected_chapter_num}" in marker.lower() or f"letter {selected_chapter_num}" in marker.lower():
                chapter_start = idx + 1  # start after the chapter marker
                break

        # Find the start of the next chapter
        chapter_end = len(self.ebook_sentences)
        for idx, marker in self.ebook_chapter_markers.items():
            if f"chapter {selected_chapter_num + 1}" in marker.lower() or f"letter {selected_chapter_num + 1}" in marker.lower():
                chapter_end = idx
                break

        # Get the sentences for this chapter
        self.chapter_sentences = self.ebook_sentences[chapter_start:chapter_end]
        self._display_chapter_text()

        # Load audio file for playback
        json_path = Path(self.audio_json_label.cget("text"))
        audio_path = json_path.parent.parent.parent / "openwhispersync" / "files" / "frankenstein" / chapter_info.get("filename")
        try:
            print(f"Attempting to load audio from: {audio_path}")
            print(f"Audio file exists: {audio_path.exists()}")
            print(f"Full path: {audio_path.absolute()}")
            self.audio_data, self.sample_rate = sf.read(str(audio_path))
            self.audio_duration = len(self.audio_data) / self.sample_rate
            self.seek_slider.config(to=self.audio_duration)
            self.current_time = 0.0
            self.update_time_label()
            print(f"Loaded audio file: {audio_path}")
        except Exception as e:
            print(f"Error loading audio file: {e}")
            self._disable_controls()
            return

        # Load alignment data
        chapter_num = chapter_info.get("filename").split("_")[1]  # Extract from filename (e.g. frankenstein_00_shelley -> 00)
        print(f"Using chapter number from filename: {chapter_num}")
        alignment_filename = f"chapter_{chapter_num}_alignment.json"
        alignment_path = Path(__file__).parent.parent / "alignments" / alignment_filename

        try:
            print(f"Looking for alignment file at: {alignment_path}")
            with open(alignment_path, 'r') as f:
                self.alignment_data = json.load(f)
            print(f"Loaded alignment data for chapter {chapter_num}")
        except FileNotFoundError:
            print(f"Alignment file not found: {alignment_path}")
            # Try without leading zero if present
            if chapter_num.startswith('0') and len(chapter_num) > 1:
                alt_chapter_num = chapter_num.lstrip('0')
                alt_alignment_filename = f"chapter_{alt_chapter_num}_alignment.json"
                alt_alignment_path = Path(__file__).parent.parent / "alignments" / alt_alignment_filename
                print(f"Trying alternative alignment file: {alt_alignment_path}")
                try:
                    with open(alt_alignment_path, 'r') as f:
                        self.alignment_data = json.load(f)
                    print(f"Loaded alignment data for chapter {alt_chapter_num}")
                except FileNotFoundError:
                    print(f"Alternative alignment file not found: {alt_alignment_path}")
                    self.alignment_data = None
            else:
                self.alignment_data = None

        # Enable controls if we have everything we need
        if self.audio_data is not None and self.chapter_sentences and self.alignment_data:
            self._enable_controls()
        else:
            self._disable_controls()
            if not self.alignment_data:
                self.text_area.config(state="normal")
                self.text_area.delete("1.0", tk.END)
                self.text_area.insert(tk.END, f"Alignment file not found for chapter {selected_chapter_num}. Please run alignment first.")
                self.text_area.config(state="disabled")

    def _display_chapter_text(self):
        self.text_area.config(state="normal")
        self.text_area.delete("1.0", tk.END)
        if self.chapter_sentences:
            # Join sentences with proper spacing
            full_text = "\n\n".join(self.chapter_sentences)
            self.text_area.insert(tk.END, full_text)
            self.text_area.tag_add("dim", "1.0", tk.END) # Dim all text initially
        self.text_area.config(state="disabled")


    def toggle_playback(self):
         if self.is_playing:
              self.pause_audio()
         else:
              self.play_audio()

    def play_audio(self):
        if self.audio_data is None or not self.alignment_data:
            print("No audio/alignment loaded to play.")
            return

        print("Playing audio...")
        self.is_playing = True
        self.play_pause_button.config(text="⏸ Pause")
        self.playback_event.set()  # Signal playback thread to start
        
        # Start playback thread
        self.playback_thread = threading.Thread(target=self._audio_playback_loop, daemon=True)
        self.playback_thread.start()
        
        # Start UI updates
        self.master.after(100, self.update_playback_ui)

    def _audio_playback_loop(self):
        """Thread function for audio playback."""
        try:
            # Calculate start sample based on current time
            start_sample = int(self.current_time * self.sample_rate)
            
            # Create output stream
            self.audio_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1 if len(self.audio_data.shape) == 1 else self.audio_data.shape[1],
                callback=self._audio_callback,
                finished_callback=self._playback_finished
            )
            
            with self.audio_stream:
                # Wait for play/pause events
                while self.is_playing and self.playback_event.is_set():
                    self.playback_event.wait()  # Blocks until event is set
                    if not self.is_playing:
                        break
                    
        except Exception as e:
            print(f"Error in playback thread: {e}")
            self.is_playing = False
            self.playback_event.clear()

    def _audio_callback(self, outdata, frames, time, status):
        """Callback for audio playback."""
        if status:
            print(f"Audio callback status: {status}")
        
        # Calculate current sample position
        current_sample = int(self.current_time * self.sample_rate)
        end_sample = min(current_sample + frames, len(self.audio_data))
        
        if current_sample >= len(self.audio_data):
            # End of audio
            outdata.fill(0)
            self.is_playing = False
            return
        
        # Get audio data for this chunk
        chunk = self.audio_data[current_sample:end_sample]
        if len(chunk) < frames:
            # Pad with zeros if we're at the end
            outdata[:len(chunk)] = chunk
            outdata[len(chunk):] = 0
        else:
            outdata[:] = chunk
        
        # Update current time
        self.current_time = end_sample / self.sample_rate

    def _playback_finished(self):
        """Called when audio playback finishes."""
        self.is_playing = False
        self.playback_event.clear()
        self.play_pause_button.config(text="▶ Play")

    def pause_audio(self):
        print("Pausing audio...")
        self.is_playing = False
        self.playback_event.clear()  # Signal playback thread to pause
        self.play_pause_button.config(text="▶ Play")
        if self.audio_stream:
            self.audio_stream.stop()

    def seek_audio(self, value):
        if not self.is_playing:  # Only update internal time if paused
            self.current_time = float(value)
            self.update_time_label()
            self._highlight_current_sentence()
        else:
            # If playing, update time and signal playback thread
            self.current_time = float(value)
            self.playback_event.set()  # Signal thread to update position

    def update_playback_ui(self):
         """Periodically updates the UI during playback."""
         if self.is_playing:
             # TODO: Get current time from playback thread
             # self.current_time = get_current_playback_time()
             self.current_time += 0.1 # Simulate time passing
             if self.current_time > self.audio_duration:
                  self.current_time = self.audio_duration
                  self.pause_audio() # End playback

             self.seek_slider.set(self.current_time)
             self.update_time_label()
             self._highlight_current_sentence()
             self.master.after(100, self.update_playback_ui) # Schedule next update

    def update_time_label(self):
        current_m, current_s = divmod(int(self.current_time), 60)
        total_m, total_s = divmod(int(self.audio_duration), 60)
        self.time_label.config(text=f"{current_m:02d}:{current_s:02d} / {total_m:02d}:{total_s:02d}")

    def _highlight_current_sentence(self):
        if not self.alignment_data or not self.chapter_sentences:
            return

        current_sentence_index = -1
        # Find the sentence that contains the current time
        for i, match in enumerate(self.alignment_data):
             # Use start time slightly before end for better sync feel
            if match['start'] <= self.current_time < match['end']:
                current_sentence_index = i
                break
        # If time is past the last match, highlight the last sentence
        if current_sentence_index == -1 and self.current_time >= self.alignment_data[-1]['end']:
             current_sentence_index = len(self.alignment_data) - 1

        if current_sentence_index != -1 and current_sentence_index < len(self.chapter_sentences):
            self.text_area.config(state="normal")
            # Remove previous highlights and dimming
            self.text_area.tag_remove("highlight", "1.0", tk.END)
            self.text_area.tag_remove("dim", "1.0", tk.END)

            # Find start and end position of the sentence in the text area
            # This is approximate and depends on how text was joined in _display_chapter_text
            target_sentence = self.chapter_sentences[current_sentence_index]
            start_pos = self.text_area.search(target_sentence, "1.0", tk.END)

            if start_pos:
                end_pos = f"{start_pos}+{len(target_sentence)}c"
                # Highlight the current sentence
                self.text_area.tag_add("highlight", start_pos, end_pos)
                # Dim text before and after
                self.text_area.tag_add("dim", "1.0", start_pos)
                self.text_area.tag_add("dim", end_pos, tk.END)
                # Scroll to make the highlighted sentence visible
                self.text_area.see(start_pos)
            else:
                 # Fallback: Dim everything if sentence not found (shouldn't happen ideally)
                 self.text_area.tag_add("dim", "1.0", tk.END)

            self.text_area.config(state="disabled")


    def _disable_controls(self):
        self.play_pause_button.config(state="disabled")
        self.seek_slider.config(state="disabled")
        self.text_area.config(state="normal")
        self.text_area.delete("1.0", tk.END)
        self.text_area.insert(tk.END, "Load audio JSON, ebook, and ensure alignment file exists.")
        self.text_area.config(state="disabled")

    def _enable_controls(self):
        self.play_pause_button.config(state="normal")
        self.seek_slider.config(state="normal")
        self.text_area.config(state="normal")
        self.text_area.delete("1.0", tk.END)
        self.text_area.insert(tk.END, "Controls enabled")
        self.text_area.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = SyncViewerApp(root)
    root.mainloop() 