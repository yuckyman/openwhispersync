import tkinter as tk
import json
from dataclasses import dataclass
from typing import List, Optional
import time
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
import os

@dataclass
class AlignmentEntry:
    sentence: str
    start_time: float
    end_time: float
    confidence: float
    matched_text: str
    is_silence_based: bool

class KaraokeVisualizer:
    def __init__(self, alignment_file: str, audio_file: str = None):
        self.root = tk.Tk()
        self.root.title("murderbot karaoke")
        
        # load and process alignment data
        with open(alignment_file, 'r') as f:
            data = json.load(f)
        
        # group entries by sentence and sort by confidence
        sentence_groups = {}
        for entry in data:
            if entry['sentence'] not in sentence_groups:
                sentence_groups[entry['sentence']] = []
            sentence_groups[entry['sentence']].append(
                AlignmentEntry(**entry)
            )
        
        # for each sentence, keep only the entry with highest confidence
        self.alignments = []
        for sentence, entries in sentence_groups.items():
            best_entry = max(entries, key=lambda x: x.confidence)
            self.alignments.append(best_entry)
        
        # sort by start time
        self.alignments.sort(key=lambda x: x.start_time)
        
        # load audio file
        self.audio_file = audio_file
        if audio_file and os.path.exists(audio_file):
            self.audio_data, self.sample_rate = sf.read(audio_file)
            if len(self.audio_data.shape) > 1:  # if stereo, convert to mono
                self.audio_data = np.mean(self.audio_data, axis=1)
        else:
            self.audio_data = None
            self.sample_rate = None
        
        # create text widget
        self.text = tk.Text(self.root, wrap=tk.WORD, font=('Courier', 14))
        self.text.pack(expand=True, fill='both', padx=20, pady=20)
        
        # insert all text
        for i, entry in enumerate(self.alignments):
            self.text.insert('end', entry.sentence + '\n')
            # create a tag for this sentence
            tag_name = f"sentence_{i}"
            self.text.tag_config(tag_name, background='lightgray')
            # apply tag to the sentence
            start = f"{i+1}.0"
            end = f"{i+1}.end"
            self.text.tag_add(tag_name, start, end)
        
        # make text widget read-only
        self.text.config(state='disabled')
        
        # create progress bar
        self.progress = tk.Scale(self.root, from_=0, 
                               to=max(a.end_time for a in self.alignments),
                               orient='horizontal',
                               command=self.seek)
        self.progress.pack(fill='x', padx=20)
        
        # create play/pause button
        self.playing = False
        self.play_button = tk.Button(self.root, text="play", command=self.toggle_play)
        self.play_button.pack(pady=10)
        
        # initialize current time and audio stream
        self.current_time = 0
        self.last_update = time.time()
        self.audio_stream = None
        self.audio_thread = None
        
        # create playback thread
        self.playback_thread = None
    
    def audio_callback(self, outdata, frames, time, status):
        """Callback for audio playback"""
        if status:
            print(status)
        
        if self.playing:
            # calculate current position in audio data
            pos = int(self.current_time * self.sample_rate)
            chunk = self.audio_data[pos:pos + frames]
            
            if len(chunk) < frames:
                outdata[:len(chunk), 0] = chunk
                outdata[len(chunk):, 0] = 0
                raise sd.CallbackStop()
            else:
                outdata[:, 0] = chunk
    
    def seek(self, value):
        """Called when user moves progress bar"""
        self.current_time = float(value)
        self.update_highlighting()
        
        # if audio is playing, restart stream at new position
        if self.playing and self.audio_stream:
            self.audio_stream.stop()
            self.start_audio_stream()
    
    def start_audio_stream(self):
        """Start audio playback from current position"""
        if self.audio_data is not None and self.sample_rate is not None:
            self.audio_stream = sd.OutputStream(
                channels=1,
                callback=self.audio_callback,
                samplerate=self.sample_rate
            )
            self.audio_stream.start()
    
    def stop_audio_stream(self):
        """Stop audio playback"""
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
    
    def toggle_play(self):
        """Toggle play/pause state"""
        self.playing = not self.playing
        self.play_button.config(text="pause" if self.playing else "play")
        
        if self.playing:
            self.last_update = time.time()
            if self.playback_thread is None or not self.playback_thread.is_alive():
                self.playback_thread = threading.Thread(target=self.playback_loop)
                self.playback_thread.daemon = True
                self.playback_thread.start()
            
            # start audio playback
            self.start_audio_stream()
        else:
            # stop audio playback
            self.stop_audio_stream()
    
    def playback_loop(self):
        """Main playback loop"""
        while self.playing:
            now = time.time()
            elapsed = now - self.last_update
            self.last_update = now
            
            self.current_time += elapsed
            self.progress.set(self.current_time)
            self.update_highlighting()
            
            # if we've reached the end, stop playing
            if self.current_time >= float(self.progress.cget('to')):
                self.playing = False
                self.play_button.config(text="play")
                self.stop_audio_stream()
                break
            
            time.sleep(0.016)  # ~60fps
    
    def update_highlighting(self):
        """Update text highlighting based on current time"""
        # reset all highlighting
        for i in range(len(self.alignments)):
            tag_name = f"sentence_{i}"
            self.text.tag_config(tag_name, background='lightgray')
        
        # find current and next sentences
        current_idx = None
        next_idx = None
        
        for i, entry in enumerate(self.alignments):
            if entry.start_time <= self.current_time <= entry.end_time:
                current_idx = i
            elif entry.start_time > self.current_time:
                next_idx = i
                break
        
        # highlight current sentence
        if current_idx is not None:
            self.text.tag_config(f"sentence_{current_idx}", background='yellow')
        
        # highlight next sentence
        if next_idx is not None:
            self.text.tag_config(f"sentence_{next_idx}", background='lightblue')
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()
        self.stop_audio_stream()  # cleanup

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python visualize_karaoke.py <alignment_file> [audio_file]")
        sys.exit(1)
    
    alignment_file = sys.argv[1]
    audio_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    visualizer = KaraokeVisualizer(alignment_file, audio_file)
    visualizer.run() 