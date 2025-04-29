import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from textwrap import wrap

def visualize_confidence(alignment_file):
    # read alignment data
    with open(alignment_file, 'r') as f:
        alignments = json.load(f)
    
    # process unique alignments (similar to previous script)
    seen = set()
    unique_alignments = []
    
    for align in alignments:
        time_key = (align['start_time'], align['end_time'])
        if time_key not in seen:
            seen.add(time_key)
            unique_alignments.append(align)
    
    # sort by confidence
    sorted_alignments = sorted(unique_alignments, key=lambda x: x['confidence'])
    confidences = [a['confidence'] * 100 for a in sorted_alignments]
    
    # create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[1, 2])
    fig.suptitle('Alignment Confidence Analysis', fontsize=16, y=0.95)
    
    # plot 1: histogram of confidence scores
    ax1.hist(confidences, bins=30, color='skyblue', edgecolor='black')
    ax1.set_title('Distribution of Confidence Scores')
    ax1.set_xlabel('Confidence Score (%)')
    ax1.set_ylabel('Number of Alignments')
    
    # add mean and median lines
    mean_conf = np.mean(confidences)
    median_conf = np.median(confidences)
    ax1.axvline(mean_conf, color='red', linestyle='--', label=f'Mean: {mean_conf:.1f}%')
    ax1.axvline(median_conf, color='green', linestyle='--', label=f'Median: {median_conf:.1f}%')
    ax1.legend()
    
    # plot 2: top and bottom 5 confidence scores with text
    num_examples = 5
    lowest = sorted_alignments[:num_examples]
    highest = sorted_alignments[-num_examples:]
    
    # prepare data for plotting
    y_positions = np.arange(num_examples * 2)
    confidences = [a['confidence'] * 100 for a in lowest + highest]
    sentences = [a['sentence'] for a in lowest + highest]
    
    # wrap long sentences
    wrapped_sentences = ['\n'.join(wrap(s, width=50)) for s in sentences]
    
    # create horizontal bar chart
    bars = ax2.barh(y_positions, confidences, color=['lightcoral']*num_examples + ['lightgreen']*num_examples)
    
    # customize the plot
    ax2.set_title('Highest and Lowest Confidence Alignments')
    ax2.set_xlabel('Confidence Score (%)')
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(wrapped_sentences)
    
    # add confidence values on the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{confidences[i]:.1f}%',
                va='center')
    
    # add a horizontal line separating low and high confidence examples
    ax2.axhline(y=num_examples-0.5, color='gray', linestyle='-', alpha=0.3)
    
    # adjust layout and display
    plt.tight_layout()
    plt.savefig('confidence_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize alignment confidence scores')
    parser.add_argument('alignment_file', help='Path to the alignment JSON file')
    args = parser.parse_args()
    
    visualize_confidence(args.alignment_file)

if __name__ == '__main__':
    main() 