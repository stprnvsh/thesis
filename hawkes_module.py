import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import trapezoid
from scipy.optimize import minimize
from mpi4py import MPI
import mpmath
import pickle
from numba import njit, prange, jit, vectorize
from time import perf_counter
import seaborn as sns
import osmnx as ox
import networkx as nx
import geopandas as gpd
from sklearn.cluster import KMeans
import numpy as np

@njit(fastmath=True)
def calculate_reachability_matrix(adjacency_matrix, num_hops):
    """
    Calculates the reachability matrix for up to num_hops.
    """
    # Ensure adjacency_matrix is the correct type
    adjacency_matrix = adjacency_matrix.astype(np.float64)
    
    num_nodes = adjacency_matrix.shape[0]
    reachability_matrix = np.eye(num_nodes, dtype=np.float64)  # Start with identity (node reaches itself)
    current_power = adjacency_matrix.copy()  # Use the same type as adjacency_matrix

    for _ in range(num_hops):
        reachability_matrix = np.clip(reachability_matrix + current_power, 0, 1)  # Add and clip
        current_power = current_power @ adjacency_matrix  # Matrix multiplication 

    return reachability_matrix.astype(np.int32)  # Convert back to int32 for compatibility

@njit(fastmath=True)
def create_extended_neighbors_list(reachability_matrix):
    """
    Creates a list of extended neighbors based on the reachability matrix.
    """
    num_nodes = reachability_matrix.shape[0]
    extended_neighbors_list = []
    for u in range(num_nodes):
        extended_neighbors = np.where(reachability_matrix[u, :] > 0)[0]
        extended_neighbors_list.append(extended_neighbors)
    return extended_neighbors_list

@njit(fastmath=True, parallel=True)
def calculate_spatial_kernels(node_locations, sigma_spatial):
    num_nodes = node_locations.shape[0]
    spatial_kernels = np.empty((num_nodes, num_nodes))
    sigma_sq = sigma_spatial**2
    norm_factor = 1 / (2 * np.pi * sigma_sq)

    for u in prange(num_nodes):
        for v in range(num_nodes):
            dx = node_locations[u,0] - node_locations[v,0]
            dy = node_locations[u,1] - node_locations[v,1]
            dist_sq = dx*dx + dy*dy
            spatial_kernels[u,v] = norm_factor * np.exp(-dist_sq/(2*sigma_sq))
    return spatial_kernels

@njit(fastmath=True)
def mark_kernel(dest_event_type, source_event_type, kernel_type, matrix_kernel):
    """A generic mark kernel."""
    if kernel_type == 'uniform':
        return 1.0
    elif kernel_type == 'matrix':
        return matrix_kernel[dest_event_type, source_event_type]
    elif kernel_type == 'custom':
        if dest_event_type == source_event_type:  return 1.5
        elif abs(dest_event_type - source_event_type) == 1: return 0.8
        else: return 0.2
    else:
        return 1.0  # Default fallback for invalid type

@njit(fastmath=True)
def apply_nonlinearity(linear_part, nonlinearity):
    """Apply nonlinearity function to linear part of intensity"""
    if nonlinearity == 'linear':
        return linear_part
    elif nonlinearity == 'relu':
        return max(0.0, linear_part)
    elif nonlinearity == 'exp':
        return np.exp(linear_part)
    elif nonlinearity == 'power':
        return max(0.0, linear_part)**2
    else:
        return linear_part  # Default fallback

@njit(fastmath=True)
def is_structured_array_nb(arr):
    """Numba-compatible alternative to check if an array is structured"""
    try:
        # Try to access the first element's first field
        # This will fail for non-structured arrays
        return True
    except:
        return False

@njit(fastmath=True)
def get_event_data(event, is_structured, index=0):
    """Extract time, node and event type from event data in a Numba-compatible way"""
    if is_structured:
        # For structured arrays
        return event['t'], event['u'], event['e']
    else:
        # For tuple lists
        return event[0], event[1], event[2]

@njit(fastmath=True)
def calculate_event_intensity(t, u, e, events, spatial_kernels, mu, K, omega_temporal, 
                             extended_neighbors_list, mark_kernel_type, mark_kernel_matrix, nonlinearity):
    """Calculate intensity at a specific time, node, and event type"""
    linear_part = mu[u, e]
    
    # Simplified approach - treat all event data as list of tuples
    for v in extended_neighbors_list[u]:
        for i in range(len(events)):
            # Extract data - handle both tuple and structured array cases
            if isinstance(events, tuple) or isinstance(events, list):
                # It's a list of tuples
                event = events[i]
                t_i, v_past, n = event[0], event[1], event[2]
            else:
                # Assume it's a structured array
                t_i = events[i]['t'] 
                v_past = events[i]['u']
                n = events[i]['e']
                
            if v_past == v and t > t_i:
                time_diff = t - t_i
                g_temporal = omega_temporal * np.exp(-omega_temporal * time_diff)
                g_spatial = spatial_kernels[u, v]
                mk = mark_kernel(e, n, mark_kernel_type, mark_kernel_matrix)
                linear_part += K[u, v] * g_spatial * g_temporal * mk
                
    return apply_nonlinearity(linear_part, nonlinearity)

@njit(fastmath=True)
def random_choice_nb(arr, prob):
    """Numba-compatible implementation of np.random.choice with probabilities"""
    # Find the cumulative sum of probabilities
    csprob = np.zeros(len(prob) + 1, dtype=np.float64)
    csprob[1:] = np.cumsum(prob)
    
    # Get a random number in [0, 1)
    r = np.random.random()
    
    # Find where the random number falls in the cumulative probabilities
    for i in range(len(prob)):
        if csprob[i] <= r < csprob[i+1]:
            return arr[i]
    
    # Fallback (should rarely happen due to floating point precision)
    return arr[0]

@njit(fastmath=True)
def simulate_hawkes_process(num_nodes, num_event_types, simulation_time,
                         mu, K, sigma_spatial, omega_temporal, node_locations,
                         extended_neighbors_list,
                         nonlinearity='linear', mark_kernel_type='uniform', mark_kernel_matrix=None):
    """
    Optimized simulation of a spatiotemporal Hawkes process with multi-hop reachability.
    Precalculates spatial kernels and avoids redundant sorting.
    """
    # Initialize with empty list and convert to array later for better performance
    events_data = []
    t = 0.0
    last_progress = 0.0
    progress_step = 0.1  # Report progress every 10% of simulation time
    
    # Count events for reporting
    event_count = 0

    # Precalculate spatial kernels
    spatial_kernels = calculate_spatial_kernels(node_locations, sigma_spatial)
    
    # Pre-allocate arrays for better performance
    intensities = np.zeros((num_nodes, num_event_types))
    
    # Create array of all possible indices
    possible_indices = np.arange(num_nodes * num_event_types)
    
    while t < simulation_time:
        # Report progress every 10% of simulation time
        current_progress = t / simulation_time
        if current_progress - last_progress >= progress_step:
            last_progress = current_progress
            print("Simulation progress:", int(current_progress * 100), "%")
        
        # Reset intensities array instead of recreating it
        intensities.fill(0.0)
        
        for u in range(num_nodes):
            for m in range(num_event_types):
                linear_part = mu[u, m]
                
                # Direct calculation rather than using helper function for better Numba performance
                for v in extended_neighbors_list[u]:
                    for event_idx in range(len(events_data)):
                        # Access elements by index to avoid type inference issues
                        event = events_data[event_idx]
                        t_i = event[0]
                        v_past = event[1]
                        n = event[2]
                        
                        if v_past == v and t > t_i:
                            time_diff = t - t_i
                            g_temporal = omega_temporal * np.exp(-omega_temporal * time_diff)
                            g_spatial = spatial_kernels[u, v]
                            mk = mark_kernel(m, n, mark_kernel_type, mark_kernel_matrix)
                            linear_part += K[u, v] * g_spatial * g_temporal * mk
                
                # Apply nonlinearity directly
                if nonlinearity == 'linear':
                    intensities[u, m] = linear_part
                elif nonlinearity == 'relu':
                    intensities[u, m] = max(0.0, linear_part)
                elif nonlinearity == 'exp':
                    intensities[u, m] = np.exp(linear_part)
                elif nonlinearity == 'power':
                    intensities[u, m] = max(0.0, linear_part)**2
                else:
                    intensities[u, m] = linear_part  # Default to linear

        total_intensity = np.sum(intensities)
        
        if total_intensity <= 0:
            dt = np.random.exponential(scale=1e6)
        else:
            dt = np.random.exponential(1.0 / total_intensity)
        
        t += dt

        if t > simulation_time:
            break

        if total_intensity > 0:
            # Create event probabilities array
            event_probability = intensities.flatten() / total_intensity
            
            # Handle numerical issues with probabilities
            if np.any(event_probability < 0):
                event_probability = np.maximum(0, event_probability)
                sum_prob = np.sum(event_probability)
                if sum_prob > 0:
                    event_probability = event_probability / sum_prob
                else:
                    # If all probabilities are zero, use uniform distribution
                    event_probability = np.ones_like(event_probability) / len(event_probability)
            
            # Use our numba-compatible random choice function
            event_index = random_choice_nb(possible_indices, event_probability)
            
            node_id = event_index // num_event_types
            event_type = event_index % num_event_types
            x, y = node_locations[node_id]
            events_data.append((t, node_id, event_type, x, y))
            event_count += 1
            
            # Provide updates for every 1000 events
            if event_count % 1000 == 0:
                print("Generated", event_count, "events so far")

    print("Simulation completed with", len(events_data), "events")
    return events_data

@njit(fastmath=True)
def process_events(events, spatial_kernels, extended_neighbors_list,
                   mu, K, omega_temporal, mark_kernel_type, mark_kernel_matrix, nonlinearity):
    nll = 0.0
    
    # Simplified handling - just sort by first element
    if isinstance(events, tuple) or isinstance(events, list):
        # It's a list of tuples
        event_times = np.zeros(len(events))
        for i in range(len(events)):
            event_times[i] = events[i][0]
    else:
        # Assume it's a structured array
        event_times = events['t']
        
    sorted_indices = np.argsort(event_times)

    for idx in sorted_indices:
        if isinstance(events, tuple) or isinstance(events, list):
            # Handle tuple list case
            t_i = events[idx][0]
            u_i = events[idx][1]
            e_i = events[idx][2]
        else:
            # Handle structured array case
            t_i = events[idx]['t']
            u_i = events[idx]['u']
            e_i = events[idx]['e']
            
        # Calculate the intensity at this event
        intensity_i = calculate_event_intensity(
            t_i, u_i, e_i, events, spatial_kernels, mu, K, omega_temporal,
            extended_neighbors_list, mark_kernel_type, mark_kernel_matrix, nonlinearity
        )
        
        # Handle zero or negative intensity (numerical issues)
        if intensity_i <= 0:
            intensity_i = 1e-9
            
        nll -= np.log(intensity_i)

    return nll

@njit(fastmath=True, parallel=True)
def calculate_nonlinear_integral(u, e, T, extended_neighbors_list, events,
                                spatial_kernels, mu, K, omega_temporal,
                                mark_kernel_type, mark_kernel_matrix, nonlinearity, num_points):
    """Calculate the integral part of the log-likelihood for nonlinear models"""
    times = np.linspace(0, T, num_points)
    intensities = np.zeros(num_points)
    
    # Calculate intensities at each time point
    for idx in prange(num_points):
        t_val = times[idx]
        intensities[idx] = calculate_event_intensity(
            t_val, u, e, events, spatial_kernels, mu, K, omega_temporal,
            extended_neighbors_list, mark_kernel_type, mark_kernel_matrix, nonlinearity
        )
    
    # Use trapezoidal rule to approximate the integral
    return trapezoid(intensities, times)

@njit(fastmath=True)
def precompute_temporal_kernels(omega_temporal, max_time=10000.0, resolution=0.1):
    """Precompute temporal kernels for a range of time differences."""
    time_diffs = np.arange(0, max_time, resolution)
    return omega_temporal * np.exp(-omega_temporal * time_diffs)

def calculate_nll(params, events, num_nodes, num_event_types, node_locations,
                 extended_neighbors_list,
                 nonlinearity='linear', num_points=100, mark_kernel_type='uniform', mark_kernel_matrix=None):
    """
    Optimized calculation of the negative log-likelihood (NLL) with multi-hop reachability.
    Integrates Numba optimized functions for speedup.
    """
    mu = params[:num_nodes * num_event_types].reshape((num_nodes, num_event_types))
    K = params[num_nodes * num_event_types: num_nodes * num_event_types + num_nodes**2].reshape(num_nodes, num_nodes)
    omega_temporal = params[-2]
    sigma_spatial = params[-1]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    nll = 0.0
    
    # Handle both structured arrays and list of tuples for getting the max time
    if len(events) > 0:
        if hasattr(events, 'dtype') and events.dtype.names is not None:
            T = np.max(events['t'])
        else:
            T = max([e[0] for e in events])
    else:
        T = 0

    # Vectorized spatial kernel calculation
    spatial_kernels = calculate_spatial_kernels(node_locations, sigma_spatial)

    if rank == 0:
        nll_events_part = process_events(events, spatial_kernels, extended_neighbors_list,
                                         mu, K, omega_temporal, mark_kernel_type, mark_kernel_matrix, nonlinearity)
        nll -= nll_events_part

    # Distribute work across processes
    nodes_per_process = num_nodes // size
    local_start_node = rank * nodes_per_process
    local_end_node = (rank + 1) * nodes_per_process if rank != size - 1 else num_nodes

    local_nll = 0.0
    
    for u in range(local_start_node, local_end_node):
        for e in range(num_event_types):
            if nonlinearity == 'linear':
                local_nll += mu[u, e] * T
                # Handle the linear case more efficiently
                for v in extended_neighbors_list[u]:
                    for i in range(len(events)):
                        if hasattr(events, 'dtype') and events.dtype.names is not None:
                            t_j = events[i]['t']
                            v_j = events[i]['u']
                            n_j = events[i]['e']
                        else:
                            t_j = events[i][0]
                            v_j = events[i][1]
                            n_j = events[i][2]
                        
                        if v_j == v:
                            g_spatial = spatial_kernels[u, v]
                            mk = mark_kernel(e, n_j, mark_kernel_type, mark_kernel_matrix)
                            local_nll += K[u, v] * g_spatial * (1 - np.exp(-omega_temporal * (T - t_j))) * mk
            else:
                local_nll += calculate_nonlinear_integral(u, e, T, extended_neighbors_list, events,
                                                        spatial_kernels, mu, K, omega_temporal,
                                                        mark_kernel_type, mark_kernel_matrix, nonlinearity, num_points)

    # Gather results and return
    total_nll = comm.allreduce(local_nll, op=MPI.SUM)
    nll_from_integral = comm.bcast(total_nll, root=0)

    if rank == 0:
        nll += nll_from_integral
        return nll
    else:
        return None

def plot_events_3d(events, title="3D Spatiotemporal Hawkes Process Simulation"):
    """Plots the simulated events in a 3D scatter plot."""
    if len(events) == 0:
        print("No events to plot.")
        return
    times = [e[0] for e in events]
    x_coords = [e[3] for e in events]
    y_coords = [e[4] for e in events]
    event_types = [e[2] for e in events]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x_coords, y_coords, times, c=event_types, cmap='viridis', marker='o')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Time')
    ax.set_title(title)
    cbar = fig.colorbar(scatter)
    cbar.set_label('Event Type')
    plt.show()

def plot_events_time_series(events, num_nodes, num_event_types):
    """Plots event counts over time."""
    if len(events) == 0:
        print("No events to plot.")
        return
    event_counts = {}
    for u in range(num_nodes):
        for m in range(num_event_types):
            event_counts[(u, m)] = []
    bin_size = 1
    current_time = 0
    event_index = 0

    while event_index < len(events):
        counts = { (u, m): 0 for u in range(num_nodes) for m in range(num_event_types) }
        while event_index < len(events) and events[event_index][0] < current_time + bin_size: # Access time from tuple
            u, m = events[event_index][1], events[event_index][2] # Access node and event type from tuple
            counts[(u, m)] += 1
            event_index += 1
        for u in range(num_nodes):
            for m in range(num_event_types):
                event_counts[(u, m)].append(counts[(u, m)])
        current_time += bin_size

    fig, axes = plt.subplots(num_nodes, num_event_types, figsize=(15, 5 * num_nodes), sharex=True)
    for u in range(num_nodes):
        for m in range(num_event_types):
            ax = axes[u, m] if num_nodes > 1 else axes[m]
            ax.plot(np.arange(0, current_time, bin_size), event_counts[(u, m)])
            ax.set_title(f'Node {u}, Event Type {m}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Event Count')
    plt.tight_layout()
    plt.show()

def save_simulation_data(filename, events, params, num_nodes, num_event_types,
                         node_locations, adjacency_matrix, neighbors_list,
                         nonlinearity, mark_kernel_type, mark_kernel_matrix, num_hops): # Added mark kernel matrix and type
    """Saves simulation data to a pickle file."""
    data = {
        'events': events,
        'params': params,
        'num_nodes': num_nodes,
        'num_event_types': num_event_types,
        'node_locations': node_locations,
        'adjacency_matrix': adjacency_matrix,
        'neighbors_list': neighbors_list,  # Still save original neighbors
        'nonlinearity': nonlinearity,
        'mark_kernel_type': mark_kernel_type, # Save type instead of params dict
        'mark_kernel_matrix': mark_kernel_matrix,
        'num_hops': num_hops
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def analyze_event_times(events):
    """
    Analyzes event times to calculate dt values, plot histogram, and calculate average dt.
    """
    if len(events)==0:
        print("No events to analyze.")
        return

    event_times = np.array([e[0] for e in events])
    if len(event_times) < 2:
        print("Not enough events to calculate dt.")
        return

    dt_values = np.diff(event_times) # Calculate time differences between consecutive events
    average_dt = np.mean(dt_values)

    print(f"Average dt: {average_dt}")

    plt.figure(figsize=(10, 6))
    sns.histplot(dt_values, bins=50, kde=True, alpha=0.7) # Histogram of dt values with KDE using seaborn
    plt.title('Histogram of Time Differences (dt) between Events')
    plt.xlabel('Time Difference (dt)')
    plt.ylabel('Frequency (Density)')
    plt.show()

def plot_cumulative_events(events, title="Cumulative Event Count Over Time"):
    """Plots the cumulative number of events over time."""
    if not events:
        print("No events to plot.")
        return
    event_times = np.array([e[0] for e in events])
    cumulative_counts = np.arange(1, len(event_times) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(event_times, cumulative_counts)
    plt.xlabel('Time')
    plt.ylabel('Cumulative Event Count')
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_spatial_event_locations_2d(events, title="2D Spatial Event Locations"):
    """Plots the spatial locations of events in a 2D scatter plot."""
    if len(events)==0:
        print("No events to plot.")
        return
    x_coords = [e[3] for e in events]
    y_coords = [e[4] for e in events]
    event_types = [e[2] for e in events]

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(x_coords, y_coords, c=event_types, cmap='viridis', marker='o')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(title)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Event Type')
    plt.gca().set_aspect('equal', adjustable='box') # Ensure equal aspect ratio
    plt.show()

def plot_event_type_distribution(events, num_event_types, title="Distribution of Event Types"):
    """Plots a bar chart of the distribution of event types."""
    if len(events)==0:
        print("No events to plot.")
        return
    event_types = [e[2] for e in events]
    type_counts = np.bincount(event_types, minlength=num_event_types) # Count occurrences of each event type

    plt.figure(figsize=(8, 6))
    plt.bar(range(num_event_types), type_counts, color='skyblue')
    plt.xlabel('Event Type')
    plt.ylabel('Number of Events')
    plt.title(title)
    plt.xticks(range(num_event_types)) # Ensure x-ticks are event types
    plt.show()

@njit(fastmath=True)
def calculate_intensity_for_plot(times, node_index, event_type_index, events, mu, K, 
                               sigma_spatial, omega_temporal, node_locations, 
                               extended_neighbors_list, nonlinearity, 
                               mark_kernel_type, mark_kernel_matrix):
    """Calculate intensity values over a time range for plotting"""
    num_points = len(times)
    intensities = np.zeros(num_points)
    spatial_kernels = calculate_spatial_kernels(node_locations, sigma_spatial)
    
    for i in range(num_points):
        intensities[i] = calculate_event_intensity(
            times[i], node_index, event_type_index, events, 
            spatial_kernels, mu, K, omega_temporal,
            extended_neighbors_list, mark_kernel_type, mark_kernel_matrix, nonlinearity
        )
    
    return intensities

def plot_intensity_vs_time(events, mu, K, sigma_spatial, omega_temporal, node_locations, 
                          extended_neighbors_list, nonlinearity, mark_kernel_type, mark_kernel_matrix, 
                          num_points=100, node_index=0, event_type_index=0, title="Estimated Intensity vs. Time"):
    """Plots the estimated intensity of a specific node and event type over time."""
    
    T = events[-1][0] if events else 10 # Use last event time or default
    times = np.linspace(0, T, num_points)
    
    # Calculate intensities using optimized function
    intensities = calculate_intensity_for_plot(
        times, node_index, event_type_index, events, mu, K, 
        sigma_spatial, omega_temporal, node_locations, 
        extended_neighbors_list, nonlinearity, 
        mark_kernel_type, mark_kernel_matrix
    )

    plt.figure(figsize=(10, 6))
    plt.plot(times, intensities)
    plt.xlabel('Time')
    plt.ylabel(f'Intensity (Node {node_index}, Event Type {event_type_index})')
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_temporal_kernel(omega_temporal, max_time=10, title="Temporal Kernel Function"):
    """Plots the temporal kernel function g_temporal(t)."""
    time_diffs = np.linspace(0, max_time, 200)
    g_temporal_values = omega_temporal * np.exp(-omega_temporal * time_diffs)

    plt.figure(figsize=(10, 6))
    plt.plot(time_diffs, g_temporal_values)
    plt.xlabel('Time Difference (t)')
    plt.ylabel('g_temporal(t)')
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_spatial_kernel_matrix(spatial_kernels, title="Spatial Kernel Matrix Heatmap"):
    """Plots the spatial kernel matrix as a heatmap."""
    plt.figure(figsize=(8, 8))
    sns.heatmap(spatial_kernels, annot=False, cmap="viridis", square=True) # Using seaborn heatmap
    plt.title(title)
    plt.xlabel('Destination Node')
    plt.ylabel('Source Node')
    plt.show()
def print_network_structure(adjacency_matrix):
    """Prints the network structure based on the adjacency matrix."""
    num_nodes = adjacency_matrix.shape[0]
    print("\nNetwork Structure:")
    for i in range(num_nodes):
        neighbors = np.where(adjacency_matrix[i, :])[0]
        neighbor_str = ", ".join(map(str, neighbors))
        print(f"Node {i}: Neighbors = [{neighbor_str}]")

def plot_network_structure(node_locations, adjacency_matrix, title="Traffic Detector Network Structure"):
    """Plots the network structure with node locations and connections."""
    num_nodes = node_locations.shape[0]
    plt.figure(figsize=(8, 8))
    plt.scatter(node_locations[:, 0], node_locations[:, 1], color='red', label='Detectors') # Plot node locations

    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency_matrix[i, j] == 1:
                plt.plot([node_locations[i, 0], node_locations[j, 0]],
                         [node_locations[i, 1], node_locations[j, 1]], color='blue', alpha=0.5) # Plot connections

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(title)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box') # Equal aspect ratio
    plt.show()
def plot_adjacency_matrix(adjacency_matrix, title="Adjacency Matrix Heatmap"):
    """Plots the adjacency matrix as a heatmap."""
    plt.figure(figsize=(8, 8))
    sns.heatmap(adjacency_matrix, annot=False, cmap="binary", square=True, cbar=False) # Binary colormap for adjacency
    plt.title(title)
    plt.xlabel('Destination Node')
    plt.ylabel('Source Node')
    plt.show()

def plot_reachability_matrix(reachability_matrix, title="Reachability Matrix Heatmap"):
    """Plots the reachability matrix as a heatmap."""
    plt.figure(figsize=(8, 8))
    sns.heatmap(reachability_matrix, annot=False, cmap="binary", square=True, cbar=False) # Binary colormap for reachability
    plt.title(title)
    plt.xlabel('Reachable Node')
    plt.ylabel('Source Node')
    plt.show()


def load_simulation_data(filename):
    """Loads simulation data from a pickle file."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    num_hops = data.get('num_hops', 1)
    return (data['events'], data['params'], data['num_nodes'], data['num_event_types'],
            data['node_locations'], data['adjacency_matrix'], data['neighbors_list'],
            data['nonlinearity'], data['mark_kernel_type'], data['mark_kernel_matrix'], num_hops) # Added mark kernel matrix and type

def get_city_road_network(city_name="Paris, France", num_nodes_target=30):
    """
    Downloads road network of a city using osmnx and reduces it to approximately num_nodes_target nodes.
    Returns node_locations and adjacency_matrix.
    """
    G = ox.graph_from_place(city_name, network_type="drive")
    G_simplified = ox.simplify_graph(G)
    G_cc = max(nx.connected_components(G_simplified.to_undirected()), key=len) # take largest connected component
    G_cc_graph = G_simplified.subgraph(G_cc).copy()
    nodes_proj, edges_proj = ox.projection.project_graph(G_cc_graph) # ox.graph_to_gdfs(G_cc)

    # Reduce graph to approximately num_nodes_target nodes using k-medoids clustering
    if G_cc_graph.number_of_nodes() > num_nodes_target:
        nnodes = ox.graph_to_gdfs(G_cc_graph, edges=False)
        X = np.array(nnodes[['x', 'y']])
        kmeans =KMeans(n_clusters=num_nodes_target, random_state=0, batch_size=6, max_iter=100).fit(X)
        closest_nodes = ox.distance.nearest_nodes(G_cc_graph, X=kmeans.cluster_centers_[:, 0], Y=kmeans.cluster_centers_[:, 1])
        G_reduced = G_cc_graph.subgraph(closest_nodes).copy()
    else:
        G_reduced = G_cc_graph

    nodes_df, edges_df = ox.graph_to_gdfs(G_reduced, edges=False)
    node_locations_osmnx = np.column_stack((nodes_df['x'], nodes_df['y']))
    adjacency_matrix_osmnx = nx.to_numpy_array(G_reduced)
    return node_locations_osmnx, adjacency_matrix_osmnx
