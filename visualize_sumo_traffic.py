#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUMO TRAFFIC DATA VISUALIZATION

This script loads and visualizes the traffic data generated from SUMO simulation.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.sparse import csr_matrix
import networkx as nx

# Load the SUMO-generated traffic data
print("Loading SUMO traffic data...")
with open('test_arbon_events_evening.pickle', 'rb') as f:
    data = pickle.load(f)

# Extract data components
events = data['events']
num_nodes = data['num_nodes']
num_event_types = data['num_event_types']
node_locations = data['node_locations']
adjacency_matrix = data['adjacency_matrix']

print(f"Data loaded successfully!")
print(f"  Events: {len(events)}")
print(f"  Nodes: {num_nodes}")
print(f"  Event types: {num_event_types}")
print(f"  Time range: {events['t'].min():.2f} - {events['t'].max():.2f} hours")

# Event type names for traffic data
event_type_names = ["Low Flow", "High Flow/Congestion"]

print(f"\n=== SUMO TRAFFIC EVENT STATISTICS ===")
print(f"Total events in simulation: {len(events)}")

if len(events) > 0:
    simulation_time = events['t'].max() - events['t'].min()
    events_per_hour = len(events) / simulation_time
    print(f"Simulation duration: {simulation_time:.1f} hours")
    print(f"Average events per hour: {events_per_hour:.1f}")
    print(f"Average events per node per hour: {events_per_hour/num_nodes:.3f}")
    
    # Event type breakdown
    event_types = events['e']
    type_counts = np.bincount(event_types, minlength=num_event_types)
    for i, count in enumerate(type_counts):
        flow_type = event_type_names[i]
        percentage = count/len(events)*100
        print(f"  {flow_type} events: {count} ({percentage:.1f}%)")
    
    # Network utilization analysis
    event_nodes = events['u']
    node_counts = np.bincount(event_nodes, minlength=num_nodes)
    active_nodes = np.sum(node_counts > 0)
    print(f"\nNetwork utilization:")
    print(f"  Active nodes: {active_nodes}/{num_nodes} ({active_nodes/num_nodes*100:.1f}%)")
    print(f"  Most active node: {np.max(node_counts)} events")
    print(f"  Least active node: {np.min(node_counts)} events")
    
    # Network connectivity analysis
    network_density = np.sum(adjacency_matrix) / (num_nodes * (num_nodes - 1))
    print(f"\nNetwork connectivity:")
    print(f"  Network density: {network_density:.3f} ({network_density*100:.1f}%)")
    print(f"  Total connections: {int(np.sum(adjacency_matrix)/2)}")

# Reachability analysis
print(f"\n=== NETWORK REACHABILITY ANALYSIS ===")

# Convert to sparse matrix for efficient computation
sparse_adj = csr_matrix(adjacency_matrix)

# Check connectivity
n_components, labels = connected_components(sparse_adj, directed=False)
print(f"Network components: {n_components}")
if n_components == 1:
    print("  Network is fully connected")
else:
    print(f"  Network has {n_components} disconnected components")

# Calculate shortest paths for reachability
try:
    distances, predecessors = shortest_path(sparse_adj, directed=False, return_predecessors=True)
    
    # Network diameter (longest shortest path)
    valid_distances = distances[distances < np.inf]
    if len(valid_distances) > 0:
        network_diameter = np.max(valid_distances)
        avg_path_length = np.mean(valid_distances)
        print(f"Network diameter: {network_diameter:.2f}")
        print(f"Average path length: {avg_path_length:.2f}")
    
    # Reachability matrix
    reachability = (distances < np.inf).astype(int)
    total_reachable_pairs = np.sum(reachability) - num_nodes  # Exclude self-reachability
    max_possible_pairs = num_nodes * (num_nodes - 1)
    reachability_ratio = total_reachable_pairs / max_possible_pairs
    print(f"Reachability ratio: {reachability_ratio:.3f} ({reachability_ratio*100:.1f}%)")
    
except Exception as e:
    print(f"Could not compute shortest paths: {e}")
    distances = None

# Generate individual visualizations
print(f"\n🎨 Generating individual visualizations...")

# 1. 3D spatio-temporal visualization
print("  Creating 3D spatio-temporal plot...")
fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_subplot(111, projection='3d')
t = events['t']
u = events['u']
e = events['e']
node_xy = np.array(node_locations)

# Color by event type
colors = ['blue', 'red']
for event_type in range(num_event_types):
    mask = (e == event_type)
    if np.any(mask):
        ax1.scatter(t[mask], node_xy[u[mask], 0], node_xy[u[mask], 1], 
                   c=colors[event_type], alpha=0.6, s=20, label=event_type_names[event_type])

ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('X coordinate')
ax1.set_zlabel('Y coordinate')
ax1.set_title('3D Spatio-Temporal Events')
ax1.legend()
plt.tight_layout()
plt.savefig('1_3d_spatiotemporal.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Time series analysis
print("  Creating time series analysis...")
fig2, ax2 = plt.subplots(figsize=(10, 6))
hours = np.floor(t).astype(int)
hour_counts = np.bincount(hours, minlength=int(np.ceil(t.max())))
ax2.bar(range(len(hour_counts)), hour_counts, alpha=0.7)
ax2.set_xlabel('Hour')
ax2.set_ylabel('Number of Events')
ax2.set_title('Events per Hour')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('2_events_per_hour.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Event timing analysis
print("  Analyzing event timing patterns...")
fig3, ax3 = plt.subplots(figsize=(10, 6))
if len(t) > 1:
    intervals = np.diff(np.sort(t))
    ax3.hist(intervals, bins=50, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Time Interval (hours)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Inter-Event Intervals')
    ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('3_inter_event_intervals.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. 2D spatial event locations
print("  Creating 2D spatial event map...")
fig4, ax4 = plt.subplots(figsize=(10, 8))
for event_type in range(num_event_types):
    mask = (e == event_type)
    if np.any(mask):
        ax4.scatter(node_xy[u[mask], 0], node_xy[u[mask], 1], 
                   c=colors[event_type], alpha=0.6, s=30, label=event_type_names[event_type])

ax4.set_xlabel('X coordinate')
ax4.set_ylabel('Y coordinate')
ax4.set_title('Spatial Event Locations')
ax4.legend()
ax4.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('4_spatial_events.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Event type distribution
print("  Creating event type distribution...")
fig5, ax5 = plt.subplots(figsize=(8, 8))
ax5.pie(type_counts, labels=event_type_names, autopct='%1.1f%%', startangle=90)
ax5.set_title('Event Type Distribution')
plt.tight_layout()
plt.savefig('5_event_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Network structure visualization
print("  Visualizing network structure...")
fig6, ax6 = plt.subplots(figsize=(10, 8))
# Plot nodes
ax6.scatter(node_xy[:, 0], node_xy[:, 1], c='lightblue', s=100, alpha=0.7, label='Nodes')

# Plot connections
for i in range(num_nodes):
    for j in range(i+1, num_nodes):
        if adjacency_matrix[i, j] > 0:
            ax6.plot([node_xy[i, 0], node_xy[j, 0]], 
                    [node_xy[i, 1], node_xy[j, 1]], 
                    'k-', alpha=0.3, linewidth=0.5)

ax6.set_xlabel('X coordinate')
ax6.set_ylabel('Y coordinate')
ax6.set_title('Network Structure')
ax6.legend()
ax6.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('6_network_structure.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Node activity heatmap
print("  Creating node activity heatmap...")
fig7, ax7 = plt.subplots(figsize=(10, 8))
node_activity = np.zeros(num_nodes)
for node in u:
    node_activity[node] += 1

# Create a scatter plot with node size proportional to activity
scatter = ax7.scatter(node_xy[:, 0], node_xy[:, 1], s=node_activity*10, 
                    c=node_activity, cmap='viridis', alpha=0.7)
ax7.set_xlabel('X coordinate')
ax7.set_ylabel('Y coordinate')
ax7.set_title('Node Activity Heatmap (size = activity level)')
plt.colorbar(scatter, label='Number of Events')
ax7.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('7_node_activity_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. Reachability visualization
if distances is not None:
    print("  Creating reachability visualization...")
    fig8, ax8 = plt.subplots(figsize=(10, 8))
    
    # Plot nodes with color based on average reachability
    avg_reachability = np.mean(reachability, axis=1)
    scatter = ax8.scatter(node_xy[:, 0], node_xy[:, 1], 
                         c=avg_reachability, cmap='plasma', s=100, alpha=0.8)
    
    # Plot network connections
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if adjacency_matrix[i, j] > 0:
                ax8.plot([node_xy[i, 0], node_xy[j, 0]], 
                        [node_xy[i, 1], node_xy[j, 1]], 
                        'k-', alpha=0.2, linewidth=0.5)
    
    ax8.set_xlabel('X coordinate')
    ax8.set_ylabel('Y coordinate')
    ax8.set_title('Node Reachability (color = avg reachability)')
    plt.colorbar(scatter, label='Average Reachability')
    ax8.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('8_node_reachability.png', dpi=300, bbox_inches='tight')
    plt.close()

# 9. Shortest path distribution
if distances is not None:
    print("  Creating shortest path distribution...")
    fig9, ax9 = plt.subplots(figsize=(10, 6))
    
    # Flatten distances and filter out infinite values
    flat_distances = distances.flatten()
    valid_distances = flat_distances[flat_distances < np.inf]
    
    if len(valid_distances) > 0:
        ax9.hist(valid_distances, bins=50, alpha=0.7, edgecolor='black')
        ax9.set_xlabel('Shortest Path Length')
        ax9.set_ylabel('Frequency')
        ax9.set_title('Distribution of Shortest Path Lengths')
        ax9.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('9_shortest_path_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

print("\n✅ All individual visualizations saved!")
print("Files created:")
print("  1_3d_spatiotemporal.png")
print("  2_events_per_hour.png") 
print("  3_inter_event_intervals.png")
print("  4_spatial_events.png")
print("  5_event_distribution.png")
print("  6_network_structure.png")
print("  7_node_activity_heatmap.png")
if distances is not None:
    print("  8_node_reachability.png")
    print("  9_shortest_path_distribution.png") 