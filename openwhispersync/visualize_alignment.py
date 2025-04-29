import json
import argparse
from rich.console import Console
from rich.table import Table
from rich.text import Text
from datetime import timedelta
from visualize import plot_alignment

def format_timestamp(seconds):
    """Convert seconds to MM:SS.mmm format"""
    td = timedelta(seconds=seconds)
    minutes = int(td.seconds / 60)
    seconds = td.seconds % 60
    milliseconds = int(td.microseconds / 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def timestamps_overlap(t1_start, t1_end, t2_start, t2_end, threshold=0.5):
    """Check if two time ranges overlap significantly"""
    # find overlap duration
    overlap_start = max(t1_start, t2_start)
    overlap_end = min(t1_end, t2_end)
    if overlap_end <= overlap_start:
        return False
    
    # calculate overlap ratio against shorter segment
    overlap_duration = overlap_end - overlap_start
    min_duration = min(t1_end - t1_start, t2_end - t2_start)
    return overlap_duration / min_duration >= threshold

def find_best_alignment(group):
    """From a group of similar alignments, pick the best one based on confidence"""
    return max(group, key=lambda x: x['confidence'])

def visualize_alignment(alignment_file):
    # create rich console
    console = Console()
    
    # read alignment data
    with open(alignment_file, 'r') as f:
        alignments = json.load(f)
    
    # create table
    table = Table(
        title="Audio-Text Alignment Visualization",
        show_header=True,
        header_style="bold magenta"
    )
    
    # add columns
    table.add_column("Time Range", justify="left", style="cyan")
    table.add_column("Sentence", style="green")
    table.add_column("Confidence", justify="right", style="yellow")
    table.add_column("Method", justify="center", style="blue")
    
    # group overlapping alignments
    processed = []
    skip = set()
    
    for i, align in enumerate(alignments):
        if i in skip:
            continue
            
        # find all similar alignments
        similar_group = [align]
        for j, other in enumerate(alignments[i+1:], start=i+1):
            if j in skip:
                continue
                
            # check if sentences are same and timestamps overlap
            if (align['sentence'] == other['sentence'] and 
                timestamps_overlap(align['start_time'], align['end_time'],
                                other['start_time'], other['end_time'])):
                similar_group.append(other)
                skip.add(j)
        
        # add best alignment from group
        if similar_group:
            processed.append(find_best_alignment(similar_group))
    
    # sort by start time
    processed.sort(key=lambda x: x['start_time'])
    
    # add rows
    for align in processed:
        time_range = f"{format_timestamp(align['start_time'])} → {format_timestamp(align['end_time'])}"
        confidence = f"{align['confidence']*100:.1f}%"
        method = "Silence" if align['is_silence_based'] else "Window"
        
        table.add_row(
            time_range,
            align['sentence'],
            confidence,
            method
        )
    
    # print table
    console.print(table)

def main():
    parser = argparse.ArgumentParser(description='Visualize alignment data')
    parser.add_argument('alignment_file', help='Path to the alignment JSON file')
    args = parser.parse_args()
    
    visualize_alignment(args.alignment_file)

# paths
audio_path = "openwhispersync/files/murderbot/All Systems Red_Part 06.mp3"
alignment_path = "openwhispersync/files/murderbot/alignments/chapter_6_alignment.json"
output_path = "openwhispersync/visualizations/murderbot/chapter_6_alignment.png"

# run visualization
plot_alignment(
    audio_path=audio_path,
    alignment_path=alignment_path,
    output_path=output_path,
    show_plot=True
)

if __name__ == '__main__':
    main() 