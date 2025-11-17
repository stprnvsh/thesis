import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Load data
data = pickle.load(open('large_arbon_events_evening_copy.pickle', 'rb'))
events = data['events']

# Extract components
times = np.array([e[0] for e in events])
nodes = np.array([e[1] for e in events])
marks = np.array([e[2] for e in events])

print(f"Total events: {len(events)}")
print(f"Time range: {times.min():.2f} to {times.max():.2f} hours")
print(f"Unique timestamps: {len(np.unique(times))}")

# Create figure
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Plot 1: Raw events timeline
ax1 = axes[0]
for i, (t, n, m) in enumerate(zip(times, nodes, marks)):
    color = 'red' if m == 0 else 'blue'
    ax1.scatter(t, n, c=color, s=20, alpha=0.7, marker='o')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Node')
ax1.set_title('Raw Events Timeline (Red=Mark 0, Blue=Mark 1)')
ax1.grid(True, alpha=0.3)

# Plot 2: Events binned by 1 hour (current model)
ax2 = axes[1]
bins_1h = np.arange(16, 23, 1)  # 16:00, 17:00, 18:00, etc.
counts_1h = np.zeros(len(bins_1h)-1)
for t in times:
    bin_idx = int(t - 16)
    if 0 <= bin_idx < len(counts_1h):
        counts_1h[bin_idx] += 1

ax2.bar(bins_1h[:-1], counts_1h, width=0.8, alpha=0.7, color='green')
ax2.set_xlabel('Time (hours)')
ax2.set_ylabel('Events per hour')
ax2.set_title('Events Binned by 1 Hour (Current Model)')
ax2.grid(True, alpha=0.3)

# Plot 3: Events binned by 15 minutes (suggested)
ax3 = axes[2]
bins_15m = np.arange(16, 22.25, 0.25)  # 16:00, 16:15, 16:30, etc.
counts_15m = np.zeros(len(bins_15m)-1)
for t in times:
    bin_idx = int((t - 16) * 4)  # 4 bins per hour
    if 0 <= bin_idx < len(counts_15m):
        counts_15m[bin_idx] += 1

ax3.bar(bins_15m[:-1], counts_15m, width=0.2, alpha=0.7, color='orange')
ax3.set_xlabel('Time (hours)')
ax3.set_ylabel('Events per 15 min')
ax3.set_title('Events Binned by 15 Minutes (Suggested)')
ax3.grid(True, alpha=0.3)

# Add statistics
stats_text = f"""Statistics:
Total events: {len(events)}
Unique timestamps: {len(np.unique(times))}
1-hour bins: {len(counts_1h)} bins, {counts_1h.sum()} events
15-min bins: {len(counts_15m)} bins, {counts_15m.sum()} events
Sparsity (1h): {(len(counts_1h) - np.count_nonzero(counts_1h)) / len(counts_1h) * 100:.1f}% empty
Sparsity (15m): {(len(counts_15m) - np.count_nonzero(counts_15m)) / len(counts_15m) * 100:.1f}% empty"""

plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('events_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nAnalysis:")
print(f"1-hour bins: {len(counts_1h)} bins, {counts_1h.sum()} events")
print(f"  Empty bins: {np.count_nonzero(counts_1h == 0)} out of {len(counts_1h)}")
print(f"  Sparsity: {(len(counts_1h) - np.count_nonzero(counts_1h)) / len(counts_1h) * 100:.1f}%")

print(f"\n15-minute bins: {len(counts_15m)} bins, {counts_15m.sum()} events")
print(f"  Empty bins: {np.count_nonzero(counts_15m == 0)} out of {len(counts_15m)}")
print(f"  Sparsity: {(len(counts_15m) - np.count_nonzero(counts_15m)) / len(counts_15m) * 100:.1f}%")

print(f"\nRecommendation: Use 15-minute bins instead of 1-hour bins!")
print(f"  This will reduce sparsity from {(len(counts_1h) - np.count_nonzero(counts_1h)) / len(counts_1h) * 100:.1f}% to {(len(counts_15m) - np.count_nonzero(counts_15m)) / len(counts_15m) * 100:.1f}%") 