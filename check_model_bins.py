import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load inference results
res = pickle.load(open('inference_result_np_large_arbon_events_evening_copy_linear.pickle', 'rb'))
print(f"Model parameters:")
print(f"  T: {res.get('T', 'Not found')}")
print(f"  N: {res.get('N', 'Not found')}")
print(f"  M: {res.get('M', 'Not found')}")

# Load data
data = pickle.load(open('large_arbon_events_evening_copy.pickle', 'rb'))
events = data['events']

# Extract components
times = np.array([e[0] for e in events])
nodes = np.array([e[1] for e in events])
marks = np.array([e[2] for e in events])

print(f"\nRaw data:")
print(f"  Total events: {len(events)}")
print(f"  Time range: {times.min():.2f} to {times.max():.2f} hours")
print(f"  Unique timestamps: {len(np.unique(times))}")

# Check if there's a window parameter that affects binning
window = res.get('window', None)
print(f"\nWindow parameter: {window}")

# Let's see what the actual observed counts look like
# The model might be using a different binning strategy
if 'obs_counts' in data:
    obs_counts = data['obs_counts']
    print(f"\nObserved counts shape: {obs_counts.shape}")
    print(f"Observed counts sample:")
    print(obs_counts[:5, :, :])
    
    # Plot observed counts
    plt.figure(figsize=(15, 8))
    
    # Plot for each node-mark combination
    for node in range(min(3, obs_counts.shape[1])):  # Show first 3 nodes
        for mark in range(obs_counts.shape[2]):
            plt.subplot(3, 2, node*2 + mark + 1)
            plt.plot(obs_counts[:, node, mark], 'o-', label=f'Node {node}, Mark {mark}')
            plt.xlabel('Time bin index')
            plt.ylabel('Event count')
            plt.title(f'Node {node}, Mark {mark}')
            plt.grid(True, alpha=0.3)
            plt.legend()
    
    plt.tight_layout()
    plt.savefig('observed_counts_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Check sparsity in observed counts
    total_bins = obs_counts.size
    non_zero_bins = np.count_nonzero(obs_counts)
    sparsity = (total_bins - non_zero_bins) / total_bins * 100
    print(f"\nObserved counts sparsity: {sparsity:.1f}%")
    print(f"Total bins: {total_bins}, Non-zero bins: {non_zero_bins}")
    
else:
    print("\nNo 'obs_counts' found in data. Let's check what keys are available:")
    print(list(data.keys()))
    
    # Try to understand the binning by looking at the time distribution
    print(f"\nTime distribution analysis:")
    time_diffs = np.diff(np.sort(times))
    print(f"  Min time difference: {time_diffs.min():.4f} hours")
    print(f"  Max time difference: {time_diffs.max():.4f} hours")
    print(f"  Mean time difference: {time_diffs.mean():.4f} hours")
    print(f"  Median time difference: {np.median(time_diffs):.4f} hours")
    
    # Check if events are clustered in time
    print(f"\nTime clustering:")
    for hour in range(16, 23):
        hour_events = times[(times >= hour) & (times < hour + 1)]
        print(f"  Hour {hour}: {len(hour_events)} events") 