#!/usr/bin/env python3
"""
Bayesian Diagnostics for Hawkes Process Inference Results

This script loads MCMC state files (.npz) and computes proper Bayesian diagnostics:
- R-hat (Gelman-Rubin statistic)
- Effective sample size (ESS)
- Trace plots
- Parameter summaries with credible intervals
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import argparse

def compute_rhat(samples):
    """Compute R-hat (Gelman-Rubin statistic) for MCMC samples"""
    if samples.ndim < 2:
        return np.nan
    
    # Reshape to (chains, samples_per_chain, ...)
    if samples.ndim == 2:
        # Assume first dimension is samples, second is parameters
        samples = samples.reshape(-1, samples.shape[1])
        n_chains = 1
        samples_per_chain = samples.shape[0]
    else:
        # Assume first dimension is chains
        n_chains = samples.shape[0]
        samples_per_chain = samples.shape[1]
    
    if n_chains == 1:
        return np.nan
    
    # Compute between-chain variance
    chain_means = np.mean(samples, axis=1)  # (chains, ...)
    overall_mean = np.mean(chain_means, axis=0)  # (...)
    B = samples_per_chain * np.var(chain_means, axis=0, ddof=1)  # (...)
    
    # Compute within-chain variance
    chain_vars = np.var(samples, axis=1, ddof=1)  # (chains, ...)
    W = np.mean(chain_vars, axis=0)  # (...)
    
    # Compute R-hat
    var_plus = (samples_per_chain - 1) / samples_per_chain * W + 1 / samples_per_chain * B
    rhat = np.sqrt(var_plus / W)
    
    return rhat

def compute_ess(samples):
    """Compute effective sample size using autocorrelation"""
    if samples.ndim < 2:
        return np.nan
    
    # For multi-dimensional arrays, compute ESS for each parameter
    if samples.ndim > 2:
        # Flatten all but first two dimensions
        original_shape = samples.shape
        samples_flat = samples.reshape(original_shape[0], -1)
        ess_flat = compute_ess(samples_flat)
        return ess_flat.reshape(original_shape[2:])
    
    # Compute autocorrelation for each parameter
    ess_list = []
    for param_idx in range(samples.shape[1]):
        param_samples = samples[:, param_idx]
        
        # Compute autocorrelation
        acf = np.correlate(param_samples, param_samples, mode='full')
        acf = acf[acf.size//2:]  # Take positive lags only
        
        # Normalize
        acf = acf / acf[0]
        
        # Find first crossing of 0.05 (Monte Carlo standard error threshold)
        threshold = 0.05
        first_crossing = np.where(acf < threshold)[0]
        if len(first_crossing) > 0:
            lag = first_crossing[0]
        else:
            lag = len(acf) - 1
        
        # ESS = N / (1 + 2*sum(autocorrelations))
        ess = len(param_samples) / (1 + 2 * np.sum(acf[1:lag+1]))
        ess_list.append(ess)
    
    return np.array(ess_list)

def plot_trace_plots(samples, param_names, save_path=None):
    """Create trace plots for MCMC samples"""
    n_params = len(param_names)
    n_chains = samples.shape[0] if samples.ndim > 2 else 1
    
    # Determine subplot layout
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_params == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (param_name, samples_i) in enumerate(zip(param_names, samples.T)):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        if n_chains > 1:
            # Plot each chain separately
            for chain_idx in range(n_chains):
                ax.plot(samples[chain_idx, :, i], alpha=0.7, label=f'Chain {chain_idx+1}')
            ax.legend()
        else:
            # Single chain
            ax.plot(samples_i, alpha=0.7)
        
        ax.set_title(f'{param_name}')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_params, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trace plots saved to {save_path}")
    plt.show()

def load_mcmc_state(state_file):
    """Load MCMC state file and return samples"""
    print(f"Loading MCMC state from {state_file}")
    
    try:
        state = np.load(state_file, allow_pickle=True)
        print(f"Available arrays: {list(state.keys())}")
        
        # Extract key parameters
        samples = {}
        for key in ['mu', 'K_masked', 'M_K', 'alpha', 'a_uncon', 'b_uncon']:
            if key in state:
                samples[key] = state[key]
                print(f"{key}: {state[key].shape}")
        
        # Check for quadratic parameters
        for key in ['gamma', 'q_uncon']:
            if key in state:
                samples[key] = state[key]
                print(f"{key}: {state[key].shape}")
        
        return samples, state
        
    except Exception as e:
        print(f"Error loading {state_file}: {e}")
        return None, None

def compute_diagnostics(samples_dict):
    """Compute diagnostics for all parameters"""
    diagnostics = {}
    
    for param_name, samples in samples_dict.items():
        print(f"\n=== {param_name} ===")
        
        # Reshape samples for diagnostics
        if samples.ndim == 3:  # (chains, samples, ...)
            n_chains, n_samples = samples.shape[:2]
            samples_flat = samples.reshape(n_chains * n_samples, -1)
        else:
            samples_flat = samples
            n_chains = 1
            n_samples = len(samples)
        
        print(f"Shape: {samples.shape}")
        print(f"Total samples: {n_chains * n_samples}")
        
        # Compute R-hat
        rhat = compute_rhat(samples)
        if not np.isnan(rhat).all():
            print(f"R-hat: {np.mean(rhat):.3f} (mean), {np.min(rhat):.3f} (min), {np.max(rhat):.3f} (max)")
        
        # Compute ESS
        ess = compute_ess(samples)
        if not np.isnan(ess).all():
            print(f"ESS: {np.mean(ess):.1f} (mean), {np.min(ess):.1f} (min), {np.max(ess):.1f} (max)")
        
        # Store diagnostics
        diagnostics[param_name] = {
            'rhat': rhat,
            'ess': ess,
            'samples': samples,
            'mean': np.mean(samples, axis=0),
            'std': np.std(samples, axis=0),
            'quantiles': np.percentile(samples, [2.5, 25, 50, 75, 97.5], axis=0)
        }
    
    return diagnostics

def main():
    parser = argparse.ArgumentParser(description="Compute Bayesian diagnostics for Hawkes inference results")
    parser.add_argument("--state_file", type=str, required=True, 
                       help="Path to MCMC state file (.npz)")
    parser.add_argument("--result_file", type=str, default=None,
                       help="Path to inference result file (.pickle) for comparison")
    parser.add_argument("--save_dir", type=str, default="diagnostics",
                       help="Directory to save diagnostic plots")
    parser.add_argument("--plot_traces", action="store_true", 
                       help="Generate trace plots")
    
    args = parser.parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Load MCMC state
    samples_dict, state = load_mcmc_state(args.state_file)
    if samples_dict is None:
        print("Failed to load MCMC state. Exiting.")
        return
    
    # Compute diagnostics
    diagnostics = compute_diagnostics(samples_dict)
    
    # Generate trace plots if requested
    if args.plot_traces:
        for param_name, samples in samples_dict.items():
            if samples.ndim >= 2:
                # Create trace plots for each parameter
                trace_save_path = save_dir / f"trace_{param_name}.png"
                plot_trace_plots(samples, [f"{param_name}_{i}" for i in range(samples.shape[-1])], 
                               trace_save_path)
    
    # Save diagnostics summary
    summary_path = save_dir / "diagnostics_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Bayesian Diagnostics Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for param_name, diag in diagnostics.items():
            f.write(f"{param_name}:\n")
            f.write(f"  R-hat: {np.mean(diag['rhat']):.3f} (mean)\n")
            f.write(f"  ESS: {np.mean(diag['ess']):.1f} (mean)\n")
            f.write(f"  Mean: {np.mean(diag['mean']):.6f}\n")
            f.write(f"  Std: {np.mean(diag['std']):.6f}\n")
            f.write(f"  95% CI: [{np.percentile(diag['samples'], 2.5):.6f}, {np.percentile(diag['samples'], 97.5):.6f}]\n\n")
    
    print(f"\nDiagnostics summary saved to {summary_path}")
    
    # Load and compare with inference results if provided
    if args.result_file and Path(args.result_file).exists():
        print(f"\nComparing with inference results from {args.result_file}")
        with open(args.result_file, 'rb') as f:
            results = pickle.load(f)
        
        print("Inference results keys:", list(results.keys()))
        
        # Compare posterior means
        for param_name in ['mu_hat', 'K_hat', 'M_K_hat', 'alpha_hat']:
            if param_name in results and param_name.replace('_hat', '') in diagnostics:
                true_param = param_name.replace('_hat', '')
                if true_param in diagnostics:
                    mcmc_mean = np.mean(diagnostics[true_param]['samples'], axis=0)
                    result_mean = results[param_name]
                    
                    # Compute relative difference
                    rel_diff = np.abs(mcmc_mean - result_mean) / (np.abs(result_mean) + 1e-8)
                    print(f"{param_name} relative diff: {np.mean(rel_diff):.6f}")

if __name__ == "__main__":
    main() 