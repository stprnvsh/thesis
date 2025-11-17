import pickle
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpy.linalg import matrix_power
from time import perf_counter
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
numpyro.set_platform("cpu")
numpyro.set_host_device_count(10)

# --- Helper Function: Load Data and Pre-process ---

def load_and_prepare_data(pickle_filename):
    """
    Loads data from the simulation pickle file and prepares it for inference.
    """
    print(f"--- Loading data from '{pickle_filename}' ---")
    with open(pickle_filename, 'rb') as f:
        data = pickle.load(f)

    # Extract data
    events = data['events']
    num_nodes = data['num_nodes']
    num_event_types = data['num_event_types']
    node_locations = data['node_locations']
    adjacency_matrix = data['adjacency_matrix']
    num_hops = data['num_hops']
    
    # The simulation end time T is crucial for the likelihood calculation.
    # It was not explicitly saved, so we use the time of the last event.
    T_observation = np.max(events['t']) if len(events) > 0 else 0.0

    print(f"Loaded {len(events)} events from a simulation with {num_nodes} nodes and {num_event_types} types.")
    print(f"Observation window T = {T_observation:.2f}")

    # --- Calculate Reachability Matrix ---
    # This mask ensures interactions only happen between nodes reachable within num_hops.
    # It must be identical to the one used in the generation script.
    print(f"Calculating reachability matrix for {num_hops} hop(s)...")
    adj_plus_identity = adjacency_matrix + np.identity(num_nodes)
    reachability_matrix = (matrix_power(adj_plus_identity, num_hops) > 0).astype(int)
    
    # Unpack structured event array into separate arrays
    event_times = np.array(events['t'])
    event_nodes = np.array(events['u'])
    event_marks = np.array(events['e'])

    # --- Convert to JAX arrays for use in the model ---
    prepared_data = {
        "event_times": jnp.array(event_times),
        "event_nodes": jnp.array(event_nodes),
        "event_marks": jnp.array(event_marks),
        "node_locations": jnp.array(node_locations),
        "reachability_mask": jnp.array(reachability_matrix),
        "T_observation": T_observation,
        "N": num_nodes,
        "M": num_event_types,
    }
    
    print("--- Data preparation complete ---")
    return prepared_data

# --- JAX Implementation of Kernel Functions ---

@jax.jit
def temporal_kernel(t, omega):
    """Exponential temporal decay kernel."""
    return omega * jnp.exp(-omega * t)

@jax.jit
def spatial_kernel(x_u, x_v, sigma):
    """Gaussian spatial kernel based on node locations."""
    # Add a small epsilon to sigma to prevent division by zero if sigma is ever zero
    safe_sigma = jnp.maximum(sigma, 1e-9)
    return (1 / (2 * jnp.pi * safe_sigma**2)) * jnp.exp(-jnp.sum((x_u - x_v)**2) / (2 * safe_sigma**2))

# --- JAX Implementation of the Log-Likelihood ---

def hawkes_log_likelihood(
    event_times, event_nodes, event_marks,
    mu, K, omega, sigma, M_k,
    node_locations, reachability_mask, T_observation
):
    """
    Calculates the log-likelihood for a linear spatio-temporal Hawkes process.
    This function is designed to be JIT-compiled by JAX.
    """
    n_events = len(event_times)
    n_nodes, n_marks = mu.shape

    # Pre-compute the full spatial kernel matrix (NxN)
    v_spatial_kernel = jax.vmap(lambda u: jax.vmap(lambda v: spatial_kernel(u, v, sigma))(node_locations))
    kappa_matrix = v_spatial_kernel(node_locations)

    # --- 1. Event Term: Sum of log-intensities at event times ---
    def event_loop_body(carry, i):
        log_lik_sum = carry
        t_i, u_i, e_i = event_times[i], event_nodes[i], event_marks[i]

        # Create a mask for past events (j < i)
        past_mask = jnp.arange(n_events) < i
        
        # Time differences with past events
        dt = t_i - event_times
        
        # Calculate contributions from all past events
        g_values = temporal_kernel(dt, omega)
        m_values = M_k[e_i, event_marks]
        K_contrib = K[u_i, event_nodes]
        kappa_contrib = kappa_matrix[u_i, event_nodes]
        mask_contrib = reachability_mask[u_i, event_nodes]

        # Sum up all excitation from past events, applying the past_mask
        excitation = jnp.sum(
            past_mask * mask_contrib * K_contrib * kappa_contrib * g_values * m_values
        )

        # Total intensity for the current event
        lambda_i = mu[u_i, e_i] + excitation
        
        # Add to log-likelihood (with a safe minimum for log)
        log_lik_sum += jnp.log(jnp.maximum(lambda_i, 1e-9))
        
        return log_lik_sum, None

    # Use jax.lax.scan for an efficient, JIT-compatible loop over events
    log_event_term, _ = jax.lax.scan(event_loop_body, 0.0, jnp.arange(n_events))

    # --- 2. Integral Term (Closed-form for linear case) ---
    
    # Integral of the baseline intensity
    integral_mu = jnp.sum(mu) * T_observation

    # Integral of the excitation term, vectorized over all past events 'j'
    def integral_contribution(t_j, u_j, e_j):
        # Contribution of a single past event j to the total integral
        decay_factor = 1.0 - jnp.exp(-omega * (T_observation - t_j))
        
        # Sum of interactions over all possible future events (u, e)
        spatial_network_sum = jnp.sum(reachability_mask[:, u_j] * K[:, u_j] * kappa_matrix[:, u_j])
        mark_sum = jnp.sum(M_k[:, e_j])
        
        return decay_factor * spatial_network_sum * mark_sum

    # Use vmap to compute contributions for all events in parallel
    integral_excitation_terms = jax.vmap(integral_contribution)(event_times, event_nodes, event_marks)
    total_integral_excitation = jnp.sum(integral_excitation_terms)
    
    integral_term = integral_mu + total_integral_excitation

    return log_event_term - integral_term

# --- NumPyro Parametric Model ---

def hawkes_inference_model(data):
    """
    NumPyro model defining priors and linking them to the data via the log-likelihood.
    """
    # Unpack data dictionary
    event_times = data['event_times']
    event_nodes = data['event_nodes']
    event_marks = data['event_marks']
    node_locations = data['node_locations']
    reachability_mask = data['reachability_mask']
    T_observation = data['T_observation']
    N, M = data['N'], data['M']

    # --- Priors for the model parameters ---
    # These are our beliefs about the parameters before seeing the data.
    
    # Baseline intensities (mu): Must be positive. HalfNormal is a good choice.
    mu = numpyro.sample("mu", dist.HalfNormal(0.1).expand([N, M]).to_event(2))

    # Network interaction matrix (K): Can be positive (excitation) or negative (inhibition).
    K = numpyro.sample("K", dist.Normal(0, 0.5).expand([N, N]).to_event(2))

    # Temporal decay (omega): Must be positive.
    omega = numpyro.sample("omega", dist.Gamma(2.0, 1.0))

    # Spatial scale (sigma): Must be positive.
    sigma = numpyro.sample("sigma", dist.Gamma(2.0, 1.0))
    
    # Mark interaction matrix (M_k): Can be positive or negative.
    M_k = numpyro.sample("M_k", dist.Normal(0, 1.0).expand([M, M]).to_event(2))

    # --- Link parameters to data via the custom log-likelihood ---
    log_lik = hawkes_log_likelihood(
        event_times, event_nodes, event_marks,
        mu, K, omega, sigma, M_k,
        node_locations, reachability_mask, T_observation
    )
    
    # Use numpyro.factor to add the custom log probability to the model
    numpyro.factor("log_lik", log_lik)

# --- Main Execution Block ---

if __name__ == '__main__':
    # --- 1. Load and Prepare Data ---
    pickle_filename = "traffic_hawkes_simulation2.pickle"
    try:
        data = load_and_prepare_data(pickle_filename)
    except FileNotFoundError:
        print(f"ERROR: The file '{pickle_filename}' was not found.")
        print("Please run the 'hawkes_generate.py' script first to create the data file.")
        exit()

    # --- 2. Set up and Run MCMC Inference ---
    rng_key = jax.random.PRNGKey(42)
    kernel = NUTS(hawkes_inference_model, target_accept_prob=0.8)
    mcmc = MCMC(
        kernel,
        num_warmup=2000,
        num_samples=2000,
        num_chains=1,
        progress_bar=True,
    )

    print("\n--- Starting MCMC Inference ---")
    start_time = perf_counter()
    mcmc.run(rng_key, data)
    end_time = perf_counter()
    print(f"--- MCMC Inference Complete ({end_time - start_time:.2f} seconds) ---")

    # --- 3. Display Results ---
    print("\n--- Parameter Posterior Summary ---")
    mcmc.print_summary()

    # --- 4. (Optional) Compare with Ground Truth ---
    print("\n--- Comparison with Ground Truth (from pickle file) ---")
    with open(pickle_filename, 'rb') as f:
        true_data = pickle.load(f)
    
    # Unpack true parameters
    params_true_flat = true_data['params_init']
    N, M = data['N'], data['M']
    mu_true = params_true_flat[0 : N*M].reshape((N, M))
    K_true_flat = params_true_flat[N*M : N*M + N*N]
    K_true = K_true_flat.reshape((N, N))
    omega_true = params_true_flat[-2]
    sigma_true = params_true_flat[-1]
    M_k_true = true_data['mark_kernel_matrix']

    # Get posterior means from inference
    posterior_samples = mcmc.get_samples()
    mu_inf = posterior_samples['mu'].mean(axis=0)
    K_inf = posterior_samples['K'].mean(axis=0)
    omega_inf = posterior_samples['omega'].mean()
    sigma_inf = posterior_samples['sigma'].mean()
    M_k_inf = posterior_samples['M_k'].mean(axis=0)

    print(f"\nOmega (Temporal Decay):")
    print(f"  True: {omega_true:.4f}, Inferred Mean: {omega_inf:.4f}")
    
    print(f"\nSigma (Spatial Scale):")
    print(f"  True: {sigma_true:.4f}, Inferred Mean: {sigma_inf:.4f}")

    print(f"\nBaseline (mu) - Mean Absolute Error:")
    print(f"  MAE: {np.mean(np.abs(mu_true - mu_inf)):.6f}")

    print(f"\nInteraction Matrix (K) - Mean Absolute Error (for non-zero true values):")
    # Only compare where true K was non-zero due to reachability mask
    mask = K_true != 0
    if np.any(mask):
        mae_k = np.mean(np.abs(K_true[mask] - K_inf[mask]))
        print(f"  MAE: {mae_k:.4f}")
    else:
        print("  No non-zero entries in true K to compare.")

    print(f"\nMark Matrix (M_k) - Mean Absolute Error:")
    print(f"  MAE: {np.mean(np.abs(M_k_true - M_k_inf)):.4f}")
    print("\n--- End of Analysis ---")