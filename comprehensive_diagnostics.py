#!/usr/bin/env python3
"""
Comprehensive Bayesian Diagnostics for Hawkes Process Inference Results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle

def plot_trace_plots(samples, param_name, save_path=None):
    """Create trace plots for MCMC samples"""
    if samples.ndim < 2:
        print(f"Cannot plot traces for {param_name}: insufficient dimensions")
        return
    
    # Determine layout
    if samples.ndim == 3:  # (chains, samples, params)
        n_chains, n_samples, n_params = samples.shape
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_params):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Plot each chain separately
            for chain_idx in range(n_chains):
                ax.plot(samples[chain_idx, :, i], alpha=0.7, label=f'Chain {chain_idx+1}')
            
            ax.set_title(f'{param_name}_{i}')
            ax.set_xlabel('Sample')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_params, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
    
    elif samples.ndim == 2:  # (samples, params)
        n_samples, n_params = samples.shape
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_params):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            ax.plot(samples[:, i], alpha=0.7)
            ax.set_title(f'{param_name}_{i}')
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

def plot_parameter_summaries(samples, param_name, save_path=None):
    """Create parameter summary plots with histograms and credible intervals"""
    if samples.ndim < 2:
        print(f"Cannot plot summaries for {param_name}: insufficient dimensions")
        return
    
    # Compute statistics
    if samples.ndim == 3:  # (chains, samples, params)
        # Average across chains
        samples_avg = np.mean(samples, axis=0)  # (samples, params)
        n_samples, n_params = samples_avg.shape
    else:
        samples_avg = samples
        n_samples, n_params = samples_avg.shape
    
    # Determine layout
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_params == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_params):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        param_samples = samples_avg[:, i]
        
        # Histogram
        ax.hist(param_samples, bins=50, alpha=0.7, density=True, edgecolor='black')
        
        # Add vertical lines for quantiles
        quantiles = np.percentile(param_samples, [2.5, 25, 50, 75, 97.5])
        colors = ['red', 'orange', 'green', 'orange', 'red']
        labels = ['2.5%', '25%', '50%', '75%', '97.5%']
        
        for q, color, label in zip(quantiles, colors, labels):
            ax.axvline(q, color=color, linestyle='--', alpha=0.8, label=label)
        
        ax.set_title(f'{param_name}_{i}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_params, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Parameter summaries saved to {save_path}")
    plt.show()

def comprehensive_analysis(state_file, result_file=None, save_dir="diagnostics"):
    """Perform comprehensive analysis of MCMC results"""
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE ANALYSIS: {state_file}")
    print(f"{'='*80}")
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    try:
        # Load MCMC state
        state = np.load(state_file, allow_pickle=True)
        print(f"Available arrays: {list(state.keys())}")
        
        # Key parameters to analyze
        key_params = ['mu', 'K_masked', 'M_K', 'alpha', 'a_uncon', 'b_uncon']
        
        diagnostics_summary = {}
        
        for param_name in key_params:
            if param_name in state:
                samples = state[param_name]
                print(f"\n{'='*50}")
                print(f"ANALYZING: {param_name}")
                print(f"{'='*50}")
                
                print(f"Shape: {samples.shape}")
                
                if samples.ndim >= 2:
                    # Basic statistics
                    mean_val = np.mean(samples, axis=0)
                    std_val = np.std(samples, axis=0)
                    
                    print(f"Mean shape: {mean_val.shape}")
                    print(f"Std shape: {std_val.shape}")
                    
                    # R-hat computation
                    if samples.ndim == 3 and samples.shape[0] > 1:
                        n_chains, n_samples = samples.shape[:2]
                        
                        # Compute between-chain variance
                        chain_means = np.mean(samples, axis=1)
                        overall_mean = np.mean(chain_means, axis=0)
                        B = n_samples * np.var(chain_means, axis=0, ddof=1)
                        
                        # Compute within-chain variance
                        chain_vars = np.var(samples, axis=1, ddof=1)
                        W = np.mean(chain_vars, axis=0)
                        
                        # Compute R-hat
                        var_plus = (n_samples - 1) / n_samples * W + 1 / n_samples * B
                        rhat = np.sqrt(var_plus / W)
                        
                        print(f"R-hat: {np.mean(rhat):.3f} (mean), {np.min(rhat):.3f} (min), {np.max(rhat):.3f} (max)")
                        
                        # Convergence assessment
                        if np.max(rhat) < 1.1:
                            print("✅ GOOD CONVERGENCE: All R-hat values < 1.1")
                        elif np.max(rhat) < 1.2:
                            print("⚠️  ACCEPTABLE CONVERGENCE: All R-hat values < 1.2")
                        else:
                            print("❌ POOR CONVERGENCE: Some R-hat values > 1.2")
                        
                        diagnostics_summary[param_name] = {
                            'rhat': rhat,
                            'convergence': 'good' if np.max(rhat) < 1.1 else 'acceptable' if np.max(rhat) < 1.2 else 'poor'
                        }
                    
                    # ESS computation
                    if samples.ndim == 3:
                        # Average across chains for ESS
                        samples_avg = np.mean(samples, axis=0)
                    else:
                        samples_avg = samples
                    
                    # Simple ESS using autocorrelation
                    ess_list = []
                    for param_idx in range(samples_avg.shape[1]):
                        param_samples = samples_avg[:, param_idx]
                        
                        # Compute autocorrelation
                        acf = np.correlate(param_samples, param_samples, mode='full')
                        acf = acf[acf.size//2:]
                        acf = acf / acf[0]
                        
                        # Find first crossing of 0.05
                        threshold = 0.05
                        first_crossing = np.where(acf < threshold)[0]
                        if len(first_crossing) > 0:
                            lag = first_crossing[0]
                        else:
                            lag = min(len(acf) - 1, 50)
                        
                        # ESS = N / (1 + 2*sum(autocorrelations))
                        ess = len(param_samples) / (1 + 2 * np.sum(acf[1:lag+1]))
                        ess_list.append(ess)
                    
                    ess_array = np.array(ess_list)
                    print(f"ESS: {np.mean(ess_array):.1f} (mean), {np.min(ess_array):.1f} (min), {np.max(ess_array):.1f} (max)")
                    
                    # Store diagnostics
                    if param_name in diagnostics_summary:
                        diagnostics_summary[param_name].update({'ess': ess_array})
                    else:
                        diagnostics_summary[param_name] = {'ess': ess_array}
                    
                    # Show sample values
                    if samples.ndim == 3:
                        print(f"Sample values (first chain, first sample): {samples[0, 0, :5] if samples.shape[2] > 5 else samples[0, 0, :]}")
                    elif samples.ndim == 2:
                        print(f"Sample values (first sample): {samples[0, :5] if samples.shape[1] > 5 else samples[0, :]}")
                    
                    # Generate plots
                    trace_save_path = save_dir / f"trace_{param_name}.png"
                    plot_trace_plots(samples, param_name, trace_save_path)
                    
                    summary_save_path = save_dir / f"summary_{param_name}.png"
                    plot_parameter_summaries(samples, param_name, summary_save_path)
                
                else:
                    print(f"Scalar value: {samples}")
        
        # Check for quadratic parameters
        quad_params = ['gamma', 'q_uncon', 'wq_uncon', 'beta_q']
        for param_name in quad_params:
            if param_name in state:
                samples = state[param_name]
                print(f"\n--- {param_name} (quadratic) ---")
                print(f"Shape: {samples.shape}")
                if samples.size > 0:
                    print(f"Value: {samples}")
        
        # Save comprehensive diagnostics
        summary_path = save_dir / "comprehensive_diagnostics.txt"
        with open(summary_path, 'w') as f:
            f.write("COMPREHENSIVE BAYESIAN DIAGNOSTICS\n")
            f.write("=" * 60 + "\n\n")
            
            for param_name, diag in diagnostics_summary.items():
                f.write(f"{param_name}:\n")
                if 'rhat' in diag:
                    f.write(f"  R-hat: {np.mean(diag['rhat']):.3f} (mean), {np.min(diag['rhat']):.3f} (min), {np.max(diag['rhat']):.3f} (max)\n")
                    f.write(f"  Convergence: {diag['convergence']}\n")
                if 'ess' in diag:
                    f.write(f"  ESS: {np.mean(diag['ess']):.1f} (mean), {np.min(diag['ess']):.1f} (min), {np.max(diag['ess']):.1f} (max)\n")
                f.write("\n")
        
        print(f"\nComprehensive diagnostics saved to {summary_path}")
        
        # Compare with inference results if provided
        if result_file and Path(result_file).exists():
            print(f"\nComparing with inference results from {result_file}")
            with open(result_file, 'rb') as f:
                results = pickle.load(f)
            
            print("Inference results keys:", list(results.keys()))
            
            # Compare posterior means
            for param_name in ['mu_hat', 'K_hat', 'M_K_hat', 'alpha_hat']:
                if param_name in results and param_name.replace('_hat', '') in diagnostics_summary:
                    true_param = param_name.replace('_hat', '')
                    if true_param in state:
                        mcmc_mean = np.mean(state[true_param], axis=0)
                        result_mean = results[param_name]
                        
                        # Compute relative difference
                        rel_diff = np.abs(mcmc_mean - result_mean) / (np.abs(result_mean) + 1e-8)
                        print(f"{param_name} relative diff: {np.mean(rel_diff):.6f}")
        
        return diagnostics_summary
        
    except Exception as e:
        print(f"Error analyzing {state_file}: {e}")
        return None

def main():
    # Files to analyze
    files_to_analyze = [
        {
            "name": "np_large_arbon_events_evening_copy_linear",
            "state_file": "mcmc_state_np_large_arbon_events_evening_copy_linear.npz",
            "result_file": "inference_result_np_large_arbon_events_evening_copy_linear.pickle"
        },
        {
            "name": "np_large_arbon_events_evening_copy_base",
            "state_file": "mcmc_state_np_large_arbon_events_evening_copy.npz",
            "result_file": None  # Base file, no specific result file
        },
        {
            "name": "np_test_arbon_events_evening",
            "state_file": "mcmc_state_np_test_arbon_events_evening.npz",
            "result_file": "inference_result_np_test_arbon_events_evening.pickle"
        }
    ]
    
    for file_info in files_to_analyze:
        if Path(file_info['state_file']).exists():
            output_dir = f"diagnostics_{file_info['name']}"
            comprehensive_analysis(
                file_info['state_file'], 
                file_info['result_file'],
                output_dir
            )
        else:
            print(f"\n⚠️  State file not found: {file_info['state_file']}")

if __name__ == "__main__":
    main() 