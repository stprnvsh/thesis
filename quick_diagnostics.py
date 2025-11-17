#!/usr/bin/env python3
"""
Quick Bayesian Diagnostics - Just the essentials
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def quick_metrics(samples):
    """Compute just the key metrics"""
    if samples.ndim < 2:
        return None, None
    
    # R-hat for convergence
    if samples.ndim == 3 and samples.shape[0] > 1:
        n_chains, n_samples = samples.shape[:2]
        chain_means = np.mean(samples, axis=1)
        B = n_samples * np.var(chain_means, axis=0, ddof=1)
        chain_vars = np.var(samples, axis=1, ddof=1)
        W = np.mean(chain_vars, axis=0)
        var_plus = (n_samples - 1) / n_samples * W + 1 / n_samples * B
        rhat = np.sqrt(var_plus / W)
        rhat_mean = np.mean(rhat)
        rhat_max = np.max(rhat)
    else:
        rhat_mean = rhat_max = np.nan
    
    # Simple ESS estimate
    if samples.ndim == 3:
        samples_avg = np.mean(samples, axis=0)
    else:
        samples_avg = samples
    
    # Quick ESS using first few autocorrelations
    ess_list = []
    for i in range(min(5, samples_avg.shape[1])):  # Just first 5 params
        param_samples = samples_avg[:, i]
        acf = np.correlate(param_samples, param_samples, mode='full')
        acf = acf[acf.size//2:acf.size//2+10]  # First 10 lags
        acf = acf / acf[0]
        ess = len(param_samples) / (1 + 2 * np.sum(acf[1:]))
        ess_list.append(ess)
    
    ess_mean = np.mean(ess_list) if ess_list else np.nan
    
    return rhat_mean, rhat_max, ess_mean

def plot_simple_traces(samples, param_name, save_path=None):
    """Simple trace plots - just a few key parameters"""
    if samples.ndim < 2:
        return
    
    # Take first few parameters only
    if samples.ndim == 3:
        n_chains, n_samples, n_params = samples.shape
        n_plot = min(4, n_params)  # Plot max 4 params
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i in range(n_plot):
            ax = axes[i]
            for chain_idx in range(n_chains):
                ax.plot(samples[chain_idx, :, i], alpha=0.7, label=f'Chain {chain_idx+1}')
            ax.set_title(f'{param_name}_{i}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_plot, 4):
            axes[i].set_visible(False)
    
    elif samples.ndim == 2:
        n_samples, n_params = samples.shape
        n_plot = min(4, n_params)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i in range(n_plot):
            ax = axes[i]
            ax.plot(samples[:, i], alpha=0.7)
            ax.set_title(f'{param_name}_{i}')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_plot, 4):
            axes[i].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def quick_assessment(state_file, save_dir="quick_diagnostics"):
    """Quick assessment of MCMC fit quality"""
    print(f"\n{'='*50}")
    print(f"QUICK ASSESSMENT: {Path(state_file).name}")
    print(f"{'='*50}")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    try:
        state = np.load(state_file, allow_pickle=True)
        
        # Key parameters to check
        key_params = ['mu', 'K_masked', 'M_K', 'alpha']
        
        print("\nCONVERGENCE METRICS:")
        print("-" * 30)
        
        overall_rhat = []
        overall_ess = []
        
        for param_name in key_params:
            if param_name in state:
                samples = state[param_name]
                
                # Handle scalar parameters like alpha
                if samples.ndim == 1:
                    print(f"{param_name:12s}: Scalar parameter - no R-hat/ESS")
                    continue
                
                rhat_mean, rhat_max, ess_mean = quick_metrics(samples)
                
                if not np.isnan(rhat_mean):
                    overall_rhat.append(rhat_max)
                    print(f"{param_name:12s}: R-hat = {rhat_max:.3f} | ESS = {ess_mean:.1f}")
                    
                    # Convergence assessment
                    if rhat_max < 1.1:
                        print(f"{'':12s}  ✅ GOOD")
                    elif rhat_max < 1.2:
                        print(f"{'':12s}  ⚠️  ACCEPTABLE")
                    else:
                        print(f"{'':12s}  ❌ POOR")
                else:
                    print(f"{param_name:12s}: Single chain - no R-hat")
        
        # Overall assessment
        if overall_rhat:
            max_rhat = max(overall_rhat)
            print(f"\nOVERALL ASSESSMENT:")
            print(f"Max R-hat: {max_rhat:.3f}")
            if max_rhat < 1.1:
                print("✅ FIT LOOKS GOOD - All parameters converged")
            elif max_rhat < 1.2:
                print("⚠️  FIT ACCEPTABLE - Some parameters may need more samples")
            else:
                print("❌ FIT POOR - Consider running longer or adjusting parameters")
        
        # Generate simple plots
        print(f"\nGenerating diagnostic plots...")
        
        for param_name in key_params:
            if param_name in state:
                samples = state[param_name]
                if samples.ndim >= 2:
                    trace_save_path = save_dir / f"trace_{param_name}.png"
                    plot_simple_traces(samples, param_name, trace_save_path)
        
        print(f"Plots saved to: {save_dir}/")
        
        return max_rhat if overall_rhat else np.nan
        
    except Exception as e:
        print(f"Error analyzing {state_file}: {e}")
        return np.nan

def main():
    # Check available state files
    state_files = [
        "mcmc_state_np_large_arbon_events_evening_copy_linear.npz",
        "mcmc_state_np_large_arbon_events_evening_copy_relu.npz",
        "mcmc_state_np_large_arbon_events_evening_copy_softplus.npz",
        "mcmc_state_np3.npz",
    ]
    
    print("QUICK BAYESIAN DIAGNOSTICS")
    print("=" * 50)
    print("This gives you just the essentials:")
    print("- R-hat (convergence): < 1.1 = good, < 1.2 = acceptable")
    print("- ESS (effective samples): higher is better")
    print("- Simple trace plots for key parameters")
    
    results = {}
    
    for state_file in state_files:
        if Path(state_file).exists():
            output_dir = f"quick_diagnostics_{Path(state_file).stem}"
            max_rhat = quick_assessment(state_file, output_dir)
            results[state_file] = max_rhat
        else:
            print(f"\n⚠️  File not found: {state_file}")
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY OF ALL FITS")
    print(f"{'='*50}")
    
    for state_file, max_rhat in results.items():
        if not np.isnan(max_rhat):
            status = "✅ GOOD" if max_rhat < 1.1 else "⚠️  ACCEPTABLE" if max_rhat < 1.2 else "❌ POOR"
            print(f"{Path(state_file).name:40s}: {max_rhat:.3f} - {status}")
        else:
            print(f"{Path(state_file).name:40s}: No convergence data")

if __name__ == "__main__":
    main() 