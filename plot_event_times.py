#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple script to plot event times from a pickle file.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_event_times(pickle_path):
    """Load and plot event times from pickle file."""
    
    # Load the pickle file
    print(f"Loading events from: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract events
    events = data["events"]
    print(f"Loaded {len(events)} events")
    
    # Extract event data
    times = events["t"]  # timestamps
    nodes = events["u"]  # node IDs
    marks = events["e"]  # mark types (0=low flow, 1=congestion)
    
    print(f"Time range: {times.min():.2f} to {times.max():.2f}")
    print(f"Number of nodes: {data['num_nodes']}")
    print(f"Number of event types: {data['num_event_types']}")
    
    # Create the plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Event timeline
    ax1.scatter(times, np.arange(len(times)), c=marks, cmap='Set1', s=50, alpha=0.7)
    ax1.set_xlabel('Time (900-second units)')
    ax1.set_ylabel('Event Index')
    ax1.set_title('Event Timeline')
    ax1.grid(True, alpha=0.3)
    
    # Add time labels for reference
    time_hours = times / 4.0  # Convert 900s units to hours
    ax1_twin = ax1.twiny()
    ax1_twin.set_xlim(ax1.get_xlim())
    ax1_twin.set_xticks(times)
    ax1_twin.set_xticklabels([f'{t/4:.1f}h' for t in times], rotation=45)
    ax1_twin.set_xlabel('Time (hours)')
    
    # Plot 2: Event distribution by node
    ax2.scatter(times, nodes, c=marks, cmap='Set1', s=50, alpha=0.7)
    ax2.set_xlabel('Time (900-second units)')
    ax2.set_ylabel('Node ID')
    ax2.set_title('Events by Node')
    ax2.grid(True, alpha=0.3)
    ax2.set_yticks(range(data['num_nodes']))
    
    # Plot 3: Event type distribution
    mark_names = ['Low Flow', 'Congestion']
    mark_counts = [np.sum(marks == 0), np.sum(marks == 1)]
    colors = ['skyblue', 'salmon']
    
    bars = ax3.bar(mark_names, mark_counts, color=colors, alpha=0.7)
    ax3.set_ylabel('Number of Events')
    ax3.set_title('Event Type Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, mark_counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    output_path = f"event_times_{Path(pickle_path).stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    
    # Show plot
    plt.show()
    
    # Print summary statistics
    print(f"\n=== EVENT SUMMARY ===")
    print(f"Total events: {len(events)}")
    print(f"Time span: {times.max() - times.min():.2f} time units")
    print(f"Time span: {(times.max() - times.min())/4:.2f} hours")
    print(f"Events per hour: {len(events)/((times.max() - times.min())/4):.2f}")
    print(f"Low flow events: {mark_counts[0]}")
    print(f"Congestion events: {mark_counts[1]}")
    
    # Check for simultaneous events
    unique_times = np.unique(times)
    simultaneous_count = len(times) - len(unique_times)
    print(f"Simultaneous events: {simultaneous_count}")
    
    # Node activity
    node_activity = np.bincount(nodes, minlength=data['num_nodes'])
    print(f"Most active node: Node {np.argmax(node_activity)} ({np.max(node_activity)} events)")
    print(f"Least active node: Node {np.argmin(node_activity)} ({np.min(node_activity)} events)")
    
    # Show actual timestamps
    print(f"\n=== TIMESTAMP DETAILS ===")
    print(f"First 20 timestamps (900-second units):")
    for i, t in enumerate(times[:20]):
        hours = t / 4.0
        print(f"  Event {i}: {t:.2f} (={hours:.2f} hours)")
    
    print(f"\nLast 20 timestamps (900-second units):")
    for i, t in enumerate(times[-20:]):
        hours = t / 4.0
        print(f"  Event {len(times)-20+i}: {t:.2f} (={hours:.2f} hours)")
    
    # Show unique timestamps
    print(f"\nUnique timestamps (first 30):")
    unique_times_sorted = np.sort(unique_times)
    for i, t in enumerate(unique_times_sorted[:30]):
        hours = t / 4.0
        print(f"  {t:.2f} (={hours:.2f} hours)")
    
    if len(unique_times) > 30:
        print(f"  ... and {len(unique_times) - 30} more unique timestamps")
    
    # Show time distribution
    print(f"\n=== TIME DISTRIBUTION ===")
    time_bins = np.linspace(times.min(), times.max(), 10)
    hist, _ = np.histogram(times, bins=time_bins)
    print(f"Time distribution across 10 bins:")
    for i, (bin_start, bin_end, count) in enumerate(zip(time_bins[:-1], time_bins[1:], hist)):
        start_hours = bin_start / 4.0
        end_hours = bin_end / 4.0
        print(f"  Bin {i}: {bin_start:.1f}-{bin_end:.1f} (={start_hours:.1f}-{end_hours:.1f}h): {count} events")

if __name__ == "__main__":
    # Plot the events
    pickle_file = "test_arbon_events_evening.pickle"
    plot_event_times(pickle_file) 