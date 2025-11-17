#!/usr/bin/env python3
"""
Simple Fit Check - Just 2 metrics + 1 plot
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def check_fit(state_file):
    """Check if MCMC fit is good - just 2 metrics"""
    print(f"\n{'='*40}")
    print(f"FIT CHECK: {Path(state_file).name}")
    print(f"{'='*40}")
    
    try:
        state = np.load(state_file, allow_pickle=True)
        
        # METRIC 1: R-hat (convergence)
        rhat_values = []
        for param_name in ['mu', 'K_masked', 'M_K']:
            if param_name in state:
                samples = state[param_name]
                if samples.ndim == 3 and samples.shape[0] > 1:
                    n_chains, n_samples = samples.shape[:2]
                    chain_means = np.mean(samples, axis=1)
                    B = n_samples * np.var(chain_means, axis=0, ddof=1)
                    chain_vars = np.var(samples, axis=1, ddof=1)
                    W = np.mean(chain_vars, axis=0)
                    var_plus = (n_samples - 1) / n_samples * W + 1 / n_samples * B
                    rhat = np.sqrt(var_plus / W)
                    rhat_values.extend(rhat.flatten())
        
        max_rhat = np.max(rhat_values) if rhat_values else np.nan
        
        # METRIC 2: ESS (effective samples)
        ess_values = []
        for param_name in ['mu', 'K_masked', 'M_K']:
            if param_name in state:
                samples = state[param_name]
                if samples.ndim >= 2:
                    # Average across chains if multiple
                    if samples.ndim == 3:
                        samples_avg = np.mean(samples, axis=0)
                    else:
                        samples_avg = samples
                    
                    # Quick ESS for first few parameters
                    for i in range(min(3, samples_avg.shape[1])):
                        param_samples = samples_avg[:, i]
                        acf = np.correlate(param_samples, param_samples, mode='full')
                        acf = acf[acf.size//2:acf.size//2+5]  # First 5 lags
                        acf = acf / acf[0]
                        ess = len(param_samples) / (1 + 2 * np.sum(acf[1:]))
                        ess_values.append(ess)
        
        min_ess = np.min(ess_values) if ess_values else np.nan
        
        # Print metrics
        print(f"METRIC 1 - Max R-hat: {max_rhat:.3f}")
        print(f"METRIC 2 - Min ESS:   {min_ess:.1f}")
        
        # Assessment
        if not np.isnan(max_rhat) and not np.isnan(min_ess):
            if max_rhat < 1.1 and min_ess > 10:
                print("✅ FIT: EXCELLENT")
            elif max_rhat < 1.2 and min_ess > 5:
                print("✅ FIT: GOOD")
            elif max_rhat < 1.3 and min_ess > 2:
                print("⚠️  FIT: ACCEPTABLE")
            else:
                print("❌ FIT: POOR")
        else:
            print("❓ FIT: Cannot assess")
        
        return max_rhat, min_ess
        
    except Exception as e:
        print(f"Error: {e}")
        return np.nan, np.nan

def plot_fit_summary(state_file, save_path=None):
    """One simple plot showing fit quality"""
    try:
        state = np.load(state_file, allow_pickle=True)
        
        # Get parameter means and stds
        params_data = {}
        for param_name in ['mu', 'K_masked', 'M_K']:
            if param_name in state:
                samples = state[param_name]
                if samples.ndim >= 2:
                    mean_val = np.mean(samples, axis=0)
                    std_val = np.std(samples, axis=0)
                    params_data[param_name] = (mean_val, std_val)
        
        # Create simple summary plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (param_name, (mean_val, std_val)) in enumerate(params_data.items()):
            ax = axes[i]
            
            # Plot mean ± std as error bars
            if mean_val.ndim == 1:
                x = np.arange(len(mean_val))
                ax.errorbar(x, mean_val, yerr=std_val, fmt='o', capsize=5)
                ax.set_title(f'{param_name} (mean ± std)')
                ax.set_xlabel('Parameter index')
                ax.set_ylabel('Value')
            else:
                # For 2D arrays, show heatmap
                im = ax.imshow(mean_val, cmap='viridis')
                ax.set_title(f'{param_name} (mean)')
                plt.colorbar(im, ax=ax)
            
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error plotting: {e}")

def main():
    # Check these files
    state_files = [
        "mcmc_state_np_large_arbon_events_evening_copy_linear.npz",
        "mcmc_state_np_large_arbon_events_evening_copy_relu.npz",
        "mcmc_state_np_large_arbon_events_evening_copy_softplus.npz",
        "mcmc_state_np3.npz",
    ]
    
    print("SIMPLE FIT CHECK")
    print("=" * 40)
    print("Just 2 metrics + 1 plot per fit")
    print("R-hat < 1.1 = good convergence")
    print("ESS > 10 = good sampling")
    
    results = {}
    
    for state_file in state_files:
        if Path(state_file).exists():
            max_rhat, min_ess = check_fit(state_file)
            results[state_file] = (max_rhat, min_ess)
            
            # Generate plot
            plot_save_path = f"fit_summary_{Path(state_file).stem}.png"
            plot_fit_summary(state_file, plot_save_path)
        else:
            print(f"\n⚠️  File not found: {state_file}")
    
    # Final summary
    print(f"\n{'='*40}")
    print("FINAL SUMMARY")
    print(f"{'='*40}")
    
    for state_file, (max_rhat, min_ess) in results.items():
        if not np.isnan(max_rhat) and not np.isnan(min_ess):
            status = "EXCELLENT" if max_rhat < 1.1 and min_ess > 10 else \
                     "GOOD" if max_rhat < 1.2 and min_ess > 5 else \
                     "ACCEPTABLE" if max_rhat < 1.3 and min_ess > 2 else "POOR"
            print(f"{Path(state_file).name:35s}: {status}")

if __name__ == "__main__":
    main() 