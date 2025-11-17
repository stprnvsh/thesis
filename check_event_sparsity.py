import pickle
import numpy as np

# Load data
data = pickle.load(open('large_arbon_events_evening_copy.pickle', 'rb'))
events = data['events']

print(f"Data structure:")
print(f"  Type: {type(events)}")
print(f"  Shape: {events.shape if hasattr(events, 'shape') else 'No shape'}")
print(f"  Length: {len(events)}")

# Extract individual components
times = np.array([e[0] for e in events])
nodes = np.array([e[1] for e in events])
marks = np.array([e[2] for e in events])
x_coords = np.array([e[3] for e in events])
y_coords = np.array([e[4] for e in events])

print(f"\nTotal events: {len(events)}")
print(f"Time range: {times.min():.2f} to {times.max():.2f} hours")
print(f"Unique timestamps: {len(np.unique(times))}")

# Count events per timestamp
unique_times, counts = np.unique(times, return_counts=True)
print(f"\nEvents per timestamp:")
print(f"  Min: {counts.min()}, Max: {counts.max()}, Mean: {counts.mean():.1f}")

print(f"\nMost common timestamps:")
for i in range(min(10, len(unique_times))):
    print(f"  {unique_times[i]:.2f}h: {counts[i]} events")

# Check sparsity
total_bins = int(times.max() - times.min()) + 1
filled_bins = len(unique_times)
sparsity = (total_bins - filled_bins) / total_bins * 100
print(f"\nSparsity: {sparsity:.1f}% of time bins are empty")

# Check mark distribution
unique_marks, mark_counts = np.unique(marks, return_counts=True)
print(f"\nMark distribution:")
for mark, count in zip(unique_marks, mark_counts):
    print(f"  Mark {mark}: {count} events ({count/len(events)*100:.1f}%)")

# Check node distribution
unique_nodes, node_counts = np.unique(nodes, return_counts=True)
print(f"\nNode distribution:")
for node, count in zip(unique_nodes, node_counts):
    print(f"  Node {node}: {count} events ({count/len(events)*100:.1f}%)")

# Show some sample events
print(f"\nSample events:")
for i in range(min(5, len(events))):
    print(f"  Event {i}: Time={times[i]:.2f}h, Node={nodes[i]}, Mark={marks[i]}, Pos=({x_coords[i]:.0f}, {y_coords[i]:.0f})") 