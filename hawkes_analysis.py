# --- In hawkes_analysis.py ---
import hawkes_module as hawkes
from optimparallel import minimize_parallel
from scipy.optimize import minimize
import numpy as np
from mpi4py import MPI

def objective_function(params, events, num_nodes, num_event_types, node_locations, extended_neighbors_list, nonlinearity, mark_kernel_type, mark_kernel_matrix): # Pass data as arguments
    return hawkes.calculate_nll(params, events, num_nodes, num_event_types, node_locations, extended_neighbors_list, nonlinearity=nonlinearity, mark_kernel_type=mark_kernel_type, mark_kernel_matrix=mark_kernel_matrix)

if __name__ == '__main__':
    pickle_filename = "traffic_hawkes_simulation.pickle"

    # Initialize MPI - moved inside __main__ block (for clarity, though not strictly necessary here)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Initialize variables - moved inside __main__ block
    events = None
    params_true_loaded = None
    num_nodes = None
    num_event_types = None
    node_locations = None
    adjacency_matrix = None
    neighbors_list = None
    nonlinearity = None
    mark_kernel_type = None
    mark_kernel_matrix = None
    num_hops = None

    if rank == 0: # Data loading only on rank 0 (inside __main__)
        (events, params_true_loaded, num_nodes, num_event_types,
         node_locations, adjacency_matrix, neighbors_list,
         nonlinearity, mark_kernel_type, mark_kernel_matrix, num_hops) = hawkes.load_simulation_data(pickle_filename)
        print(f"Rank {rank}: Data loaded")
    else:
        print(f"Rank {rank}: Preparing to receive broadcasted data...")

    # Broadcast data (inside __main__)
    events = comm.bcast(events, root=0)
    params_true_loaded = comm.bcast(params_true_loaded, root=0)
    num_nodes = comm.bcast(num_nodes, root=0)
    num_event_types = comm.bcast(num_event_types, root=0)
    node_locations = comm.bcast(node_locations, root=0)
    adjacency_matrix = comm.bcast(adjacency_matrix, root=0)
    neighbors_list = comm.bcast(neighbors_list, root=0)
    nonlinearity = comm.bcast(nonlinearity, root=0)
    mark_kernel_type = comm.bcast(mark_kernel_type, root=0)
    mark_kernel_matrix = comm.bcast(mark_kernel_matrix, root=0)
    num_hops = comm.bcast(num_hops, root=0)

    if rank != 0:
        print(f"Rank {rank}: Data received from rank 0.")

    dt = np.dtype([('t', float), ('u', int), ('e', int), ('x', float), ('y', float)])
    if events is not None:
        events = np.array(events, dtype=dt)
    if node_locations is not None:
        node_locations = np.array(node_locations)

    # Calculate reachability matrix and extended neighbors (inside __main__)
    reachability_matrix = hawkes.calculate_reachability_matrix(adjacency_matrix, num_hops)
    extended_neighbors_list = hawkes.create_extended_neighbors_list(reachability_matrix)

    # Bounds (inside __main__)
    num_mark_kernel_params = num_event_types * num_event_types
    mu_bounds = [(0.0000001, 0.1)] * (num_nodes * num_event_types)
    k_bounds = [(-1, 1)] * (num_nodes * num_nodes)
    mark_kernel_bounds = [(-0.2, 0.2)] * num_mark_kernel_params
    omega_sigma_bounds = [(0.2, 0.9), (1.5, 2.5)]
    bounds = mu_bounds + k_bounds + mark_kernel_bounds + omega_sigma_bounds

    mark_kernel_matrix = np.array([
        [0.9, 0.5, 0.2],  # From Low Flow (0) to Low, Medium, High
        [-0.5, 0.1, 0.8],  # From Medium Flow (1) to Low, Medium, High
        [0.5, 0.5, 0.9]   # From High Flow (2) to Low, Medium, High
    ])

    def generate_random_initial_params(num_nodes, num_event_types): # (inside __main__)
        """Generates random initial parameters including mark_kernel_matrix within a reasonable range."""
        mu_init = np.random.uniform(0, 0.01, size=(num_nodes, num_event_types))
        K_init = np.random.uniform(-1, 1, size=(num_nodes, num_nodes))
        mark_kernel_init = np.random.uniform(-0.2, 0.2, size=(num_event_types, num_event_types))
        omega_temporal_init = np.random.uniform(0, 1)
        sigma_spatial_init = np.random.uniform(2, 5.0)
        params_init = np.concatenate([mu_init.flatten(), K_init.flatten(), mark_kernel_init.flatten(), [omega_temporal_init, sigma_spatial_init]])
        return params_init

    params_init_random = generate_random_initial_params(num_nodes, num_event_types) # (inside __main__)
    
    if rank == 0: # Optimization - only rank 0 performs optimization (inside __main__)
        print(f"Rank {rank}: Data loaded") # Redundant print - remove
        print(f"Rank {rank}: Starting optimization from random parameters (using optimparallel)...")

        result = minimize_parallel(
            fun=objective_function,
            x0=params_init_random,
            args=(events, num_nodes, num_event_types, node_locations, extended_neighbors_list, nonlinearity, mark_kernel_type, mark_kernel_matrix), # Pass data as *args
            bounds=bounds,
            options={'maxiter': 100}
        )

        params_estimated = result.x
        mu_estimated = params_estimated[:num_nodes * num_event_types].reshape((num_nodes, num_event_types))
        K_estimated = params_estimated[num_nodes * num_event_types:num_nodes * num_event_types + num_nodes**2].reshape((num_nodes, num_nodes))
        mark_kernel_matrix_estimated = params_estimated[num_nodes * num_event_types + num_nodes**2 : num_nodes * num_event_types + num_nodes**2 + num_mark_kernel_params].reshape((num_event_types, num_event_types))
        omega_temporal_estimated = params_estimated[-2]
        sigma_spatial_estimated = params_estimated[-1]
        print(f"{result.message}, {result.x},mu_estimated:{mu_estimated}, K_estimated: {K_estimated} mark: {mark_kernel_matrix_estimated} omega tmeporal: {omega_temporal_estimated} sigma estimation:{sigma_spatial_estimated}" )

    else: # Rank != 0 does not perform optimization (inside __main__)
        pass

    comm.Barrier() # (inside __main__)
    if rank == 0: # (inside __main__)
        print(f"Rank {rank}: Analysis completed.") # (inside __main__)