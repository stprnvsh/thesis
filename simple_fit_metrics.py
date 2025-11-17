#!/usr/bin/env python3
"""
Simple Fit Metrics + PIT Plots for Hawkes Models
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import sys

def compute_ess(samples):
    """Compute Effective Sample Size for 1D array"""
    if samples.ndim > 1:
        # For multi-dimensional, average across dimensions first
        samples = np.mean(samples, axis=tuple(range(1, samples.ndim)))
    
    n = len(samples)
    if n < 2:
        return 1.0
    
    # Compute autocorrelation function
    acf = np.correlate(samples - np.mean(samples), samples - np.mean(samples), mode='full')
    acf = acf[n-1:] / acf[n-1]  # Normalize and take positive lags
    
    # Find first negative autocorrelation (cutoff)
    cutoff = np.where(acf < 0)[0]
    if len(cutoff) > 0:
        acf = acf[:cutoff[0]]
    
    # ESS = n / (1 + 2*sum(acf[1:]))
    ess = n / (1 + 2 * np.sum(acf[1:]))
    return max(ess, 1.0)

def geweke_diagnostic(samples, first=0.1, last=0.5):
    """Geweke diagnostic: compare beginning vs end of chain"""
    if samples.ndim > 1:
        # For multi-dimensional, average across dimensions first
        samples = np.mean(samples, axis=tuple(range(1, samples.ndim)))
    
    n = len(samples)
    if n < 100:
        return np.array([np.nan])
    
    first_idx = int(first * n)
    last_idx = int(last * n)
    
    if first_idx < 10 or last_idx < 10:
        return np.array([np.nan])
    
    first_part = samples[:first_idx]
    last_part = samples[-last_idx:]
    
    # Welch's t-test
    mean_diff = np.mean(first_part) - np.mean(last_part)
    var_combined = (np.var(first_part, ddof=1) / len(first_part) + 
                    np.var(last_part, ddof=1) / len(last_part))
    
    if var_combined <= 0:
        return np.array([np.nan])
    
    z_score = mean_diff / np.sqrt(var_combined)
    return np.array([z_score])

def heidelberger_welch(samples, alpha=0.05):
    """Heidelberger-Welch stationarity test"""
    if samples.ndim > 1:
        # For multi-dimensional, average across dimensions first
        samples = np.mean(samples, axis=tuple(range(1, samples.ndim)))
    
    n = len(samples)
    if n < 100:
        return np.array([np.nan])
    
    # Proper Heidelberger-Welch test: compare means of first and second half
    mid = n // 2
    first_half = samples[:mid]
    second_half = samples[mid:]
    
    # Welch's t-test for mean equality (more robust than F-test)
    mean1 = np.mean(first_half)
    mean2 = np.mean(second_half)
    var1 = np.var(first_half, ddof=1)
    var2 = np.var(second_half, ddof=1)
    
    if var1 == 0 and var2 == 0:
        # Both halves have zero variance - check if means are equal
        if np.abs(mean1 - mean2) < 1e-10:
            return np.array([1.0])  # Perfect stationarity
        else:
            return np.array([0.0])  # Different means
    
    if var1 == 0 or var2 == 0:
        return np.array([np.nan])
    
    # Welch's t-statistic
    t_stat = (mean1 - mean2) / np.sqrt(var1/len(first_half) + var2/len(second_half))
    
    # Approximate p-value using normal distribution (for large n)
    # p-value = 2 * (1 - Φ(|t|)) where Φ is normal CDF
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(np.abs(t_stat)))
    
    return np.array([p_value])

def monte_carlo_se(samples):
    """Monte Carlo Standard Error"""
    if samples.ndim > 1:
        # For multi-dimensional, average across dimensions first
        samples = np.mean(samples, axis=tuple(range(1, samples.ndim)))
    
    if np.var(samples) == 0:
        return np.array([np.nan])
    
    # MCSE = sqrt(var / ESS)
    ess = compute_ess(samples)
    mcse = np.sqrt(np.var(samples) / ess)
    return np.array([mcse])

def plot_autocorrelation(samples, param_name, ax):
    """Plot autocorrelation function for MCMC mixing assessment"""
    if samples.ndim > 1:
        # For multi-dimensional, average across dimensions first
        samples = np.mean(samples, axis=tuple(range(1, samples.ndim)))
    
    n = len(samples)
    if n < 100:
        ax.text(0.5, 0.5, f'Insufficient samples\n(n={n})', 
                ha='center', va='center', transform=ax.transAxes)
        return
    
    # Compute autocorrelation up to lag 50 or n//4
    max_lag = min(50, n // 4)
    lags = np.arange(1, max_lag + 1)
    
    acf = []
    for lag in lags:
        if lag < n:
            corr = np.corrcoef(samples[:-lag], samples[lag:])[0, 1]
            acf.append(corr if not np.isnan(corr) else 0.0)
        else:
            acf.append(0.0)
    
    acf = np.array(acf)
    
    # Plot autocorrelation
    ax.plot(lags, acf, 'b-', linewidth=1, alpha=0.8)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='±0.1 threshold')
    ax.axhline(y=-0.1, color='r', linestyle='--', alpha=0.5)
    ax.fill_between(lags, -0.1, 0.1, alpha=0.1, color='red')
    
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(f'{param_name} Autocorrelation')
    ax.grid(True, alpha=0.3)
    ax.legend()

def plot_simple_traces(samples, param_name, ax):
    """Plot simple trace plot for parameter"""
    if samples.ndim > 1:
        # For multi-dimensional, average across dimensions first
        samples = np.mean(samples, axis=tuple(range(1, samples.ndim)))
    
    ax.plot(samples, alpha=0.7, linewidth=0.5)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Value')
    ax.set_title(f'{param_name} Trace')
    ax.grid(True, alpha=0.3)

def quick_assessment(state_file):
    """Quick assessment of MCMC fit quality"""
    print(f"\n=== Analyzing {Path(state_file).name} ===")
    
    try:
        state = np.load(state_file, allow_pickle=True)
    except Exception as e:
        print(f"Error loading {state_file}: {e}")
        return
    
    # Get parameter names (excluding metadata)
    param_names = [k for k in state.keys() if k not in ['t', 'u', 'e', 'T', 'node_locations', 'reach_mask', 'start_idx', 'L_max', 'window']]
    
    print(f"Parameters: {param_names}")
    
    # Quick metrics for key parameters
    key_params = ['mu', 'K_masked', 'M_K', 'alpha']
    if 'gamma' in param_names:
        key_params.append('gamma')
    
    results = {}
    
    for param in key_params:
        if param in param_names:
            samples = state[param]
            print(f"\n--- {param} ---")
            print(f"Shape: {samples.shape}")
            
            if samples.ndim == 1:
                print("Scalar parameter - skipping R-hat/ESS")
                continue
            
            # For multi-dimensional, compute metrics on representative indices
            if samples.ndim == 3:  # (samples, dim1, dim2)
                # Test a few representative indices
                test_indices = [(0, 0), (samples.shape[1]//2, samples.shape[2]//2), (-1, -1)]
                test_samples = []
                for i, j in test_indices:
                    if i < samples.shape[1] and j < samples.shape[2]:
                        test_samples.append(samples[:, i, j])
                
                if test_samples:
                    # Use first test sample for metrics
                    test_sample = test_samples[0]
                    print(f"Testing index (0,0): {test_sample.shape}")
                    
                    geweke = geweke_diagnostic(test_sample)
                    heidel = heidelberger_welch(test_sample)
                    mcse = monte_carlo_se(test_sample)
                    
                    print(f"METRIC 1 - Geweke |Z|: {np.abs(geweke[0]):.3f} {'✅' if np.abs(geweke[0]) < 2 else '⚠️'} ({'Good' if np.abs(geweke[0]) < 2 else 'Poor'} mixing)")
                    print(f"METRIC 2 - Heidelberger p: {heidel[0]:.3f} {'✅' if heidel[0] > 0.05 else '⚠️'} ({'Good' if heidel[0] > 0.05 else 'Poor'} stationarity)")
                    print(f"METRIC 3 - Monte Carlo SE: {mcse[0]:.6f} {'✅' if mcse[0] < 0.1 else '⚠️'} ({'Good' if mcse[0] < 0.1 else 'Poor'} precision)")
                    
                    results[param] = {
                        'geweke': geweke[0],
                        'heidel': heidel[0], 
                        'mcse': mcse[0]
                    }
            else:
                print(f"Unexpected shape {samples.shape} - skipping")
    
    # Overall assessment
    if results:
        valid_metrics = 0
        good_metrics = 0
        
        for param, metrics in results.items():
            for metric_name, value in metrics.items():
                if not np.isnan(value):
                    valid_metrics += 1
                    if metric_name == 'geweke' and np.abs(value) < 2:
                        good_metrics += 1
                    elif metric_name == 'heidel' and value > 0.05:
                        good_metrics += 1
                    elif metric_name == 'mcse' and value < 0.1:
                        good_metrics += 1
        
        if valid_metrics > 0:
            quality = good_metrics / valid_metrics
            if quality >= 0.8:
                assessment = "EXCELLENT FIT"
            elif quality >= 0.6:
                assessment = "GOOD FIT"
            elif quality >= 0.4:
                assessment = "FAIR FIT"
            else:
                assessment = "POOR FIT"
            
            print(f"\n=== OVERALL ASSESSMENT ===")
            print(f"Valid metrics: {valid_metrics}")
            print(f"Good metrics: {good_metrics}")
            print(f"Quality score: {quality:.1%}")
            print(f"Assessment: {assessment}")
    
    return results

def create_diagnostic_plots(state_file, output_dir="diagnostic_plots"):
    """Create comprehensive diagnostic plots including autocorrelation"""
    print(f"\n=== Creating diagnostic plots for {Path(state_file).name} ===")
    
    try:
        state = np.load(state_file, allow_pickle=True)
    except Exception as e:
        print(f"Error loading {state_file}: {e}")
        return
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Key parameters to plot
    key_params = ['mu', 'K_masked', 'M_K', 'alpha']
    if 'gamma' in state.keys():
        key_params.append('gamma')
    
    # Create plots
    for param in key_params:
        if param in state.keys():
            samples = state[param]
            print(f"Plotting {param}...")
            
            # Skip scalar parameters or empty arrays
            if samples.ndim == 0 or samples.size == 0:
                print(f"  Skipping {param} - scalar parameter or empty")
                continue
            
            if samples.ndim == 1:
                print(f"  Skipping {param} - 1D parameter (not suitable for autocorrelation)")
                continue
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Trace plot
            plot_simple_traces(samples, param, ax1)
            
            # Autocorrelation plot
            plot_autocorrelation(samples, param, ax2)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = Path(output_dir) / f"{param}_diagnostics.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved {plot_path}")
    
    print(f"Diagnostic plots saved to {output_dir}/")

def check_divergences(state_file):
    """Check for divergent transitions in NUTS sampling"""
    print(f"\n=== Checking divergences for {Path(state_file).name} ===")
    
    try:
        state = np.load(state_file, allow_pickle=True)
    except Exception as e:
        print(f"Error loading {state_file}: {e}")
        return
    
    # Check if we have divergence information
    # Note: NumPyro doesn't always save divergence info in the same way
    # We'll check for common patterns
    
    divergence_count = 0
    divergence_info = "Unknown"
    
    # Check for explicit divergence info
    if 'diverging' in state.keys():
        divergence_count = int(np.sum(state['diverging']))
        divergence_info = f"Explicit: {divergence_count}"
    elif 'divergences' in state.keys():
        divergence_count = int(np.sum(state['divergences']))
        divergence_info = f"Explicit: {divergence_count}"
    else:
        # Check for energy diagnostics or other indicators
        if 'energy' in state.keys():
            energy = state['energy']
            if energy.ndim > 1:
                energy = np.mean(energy, axis=tuple(range(1, energy.ndim)))
            
            # Check for energy spikes (potential divergences)
            energy_diff = np.diff(energy)
            energy_spikes = np.sum(np.abs(energy_diff) > 3 * np.std(energy_diff))
            if energy_spikes > 0:
                divergence_info = f"Energy spikes detected: {energy_spikes}"
            else:
                divergence_info = "No energy spikes detected"
        else:
            divergence_info = "No divergence information available"
    
    print(f"Divergence status: {divergence_info}")
    
    # Recommendations
    if divergence_count > 0:
        print("⚠️  WARNING: Divergent transitions detected!")
        print("   - Increase target_accept_prob (e.g., 0.9)")
        print("   - Increase warmup iterations")
        print("   - Check parameter priors")
        print("   - Consider reparameterization")
    else:
        print("✅ No divergent transitions detected")
    
    return divergence_count

def compute_bfmi(state_file):
    """Compute Bayesian Fraction of Missing Information (BFMI)"""
    print(f"\n=== Computing BFMI for {Path(state_file).name} ===")
    
    try:
        state = np.load(state_file, allow_pickle=True)
    except Exception as e:
        print(f"Error loading {state_file}: {e}")
        return None
    
    # Check if we have energy information
    if 'energy' in state.keys():
        energy = state['energy']
        print(f"Energy shape: {energy.shape}")
        
        # Handle different energy formats
        if energy.ndim > 1:
            # If energy has multiple dimensions, average across them
            energy = np.mean(energy, axis=tuple(range(1, energy.ndim)))
            print(f"Averaged energy shape: {energy.shape}")
        
        n_samples = len(energy)
        if n_samples < 100:
            print(f"⚠️  Insufficient samples for reliable BFMI: {n_samples}")
            return None
        
        # Compute BFMI from energy
        # BFMI = Var(E) / Var(E_chain) where E_chain is the energy chain
        # This measures how much information is missing due to poor mixing
        
        # Remove warmup period (first 10% of samples)
        warmup_idx = max(1, n_samples // 10)
        energy_post_warmup = energy[warmup_idx:]
        
        # Compute variance of energy differences (chain variance)
        energy_diff = np.diff(energy_post_warmup)
        var_chain = np.var(energy_diff, ddof=1)
        
        # Compute variance of energy values (total variance)
        var_total = np.var(energy_post_warmup, ddof=1)
        
        if var_chain == 0:
            print("⚠️  Zero chain variance - cannot compute BFMI")
            return None
        
        # BFMI = Var(E) / Var(E_chain)
        bfmi = var_total / var_chain
        
        print(f"Energy variance (total): {var_total:.6f}")
        print(f"Energy variance (chain): {var_chain:.6f}")
        print(f"BFMI: {bfmi:.6f}")
        
    else:
        print("❌ No energy information found - cannot compute exact BFMI")
        print("   BFMI requires energy diagnostics from NUTS sampling")
        
        # Alternative: Estimate BFMI from parameter autocorrelations
        print("\n🔄 Estimating BFMI from parameter autocorrelations...")
        
        # Use key parameters to estimate mixing quality
        key_params = ['mu', 'K_masked', 'M_K']
        autocorr_scores = []
        
        for param in key_params:
            if param in state.keys():
                samples = state[param]
                if samples.ndim > 1:
                    # Test a representative element
                    test_sample = samples[:, 0, 0] if samples.ndim == 3 else samples[:, 0]
                    
                    # Compute autocorrelation at lag 1
                    if len(test_sample) > 1:
                        acf_1 = np.corrcoef(test_sample[:-1], test_sample[1:])[0, 1]
                        if not np.isnan(acf_1):
                            autocorr_scores.append(abs(acf_1))
        
        if autocorr_scores:
            # Convert autocorrelation to approximate BFMI
            # Lower autocorrelation = higher BFMI
            mean_acf = np.mean(autocorr_scores)
            estimated_bfmi = max(0.1, 1.0 - mean_acf)  # Rough approximation
            
            print(f"Mean autocorrelation (lag 1): {mean_acf:.4f}")
            print(f"Estimated BFMI: {estimated_bfmi:.4f}")
            bfmi = estimated_bfmi
        else:
            print("⚠️  Cannot estimate BFMI from parameters")
            return None
    
    # BFMI interpretation
    if bfmi >= 0.3:
        bfmi_status = "✅ EXCELLENT"
        bfmi_meaning = "Very good mixing, minimal information loss"
    elif bfmi >= 0.2:
        bfmi_status = "✅ GOOD"
        bfmi_meaning = "Good mixing, acceptable information loss"
    elif bfmi >= 0.1:
        bfmi_status = "⚠️  FAIR"
        bfmi_meaning = "Moderate mixing, some information loss"
    else:
        bfmi_status = "❌ POOR"
        bfmi_meaning = "Poor mixing, significant information loss"
    
    print(f"BFMI Status: {bfmi_status}")
    print(f"Meaning: {bfmi_meaning}")
    
    # Recommendations based on BFMI
    if bfmi < 0.3:
        print("\n🔧 RECOMMENDATIONS for low BFMI:")
        print("   - Increase warmup iterations")
        print("   - Adjust target_accept_prob")
        print("   - Check parameter priors")
        print("   - Consider reparameterization")
        print("   - Use SVI warmup before MCMC")
    
    # Instructions for enabling energy diagnostics
    if 'energy' not in state.keys():
        print("\n📋 TO ENABLE ENERGY DIAGNOSTICS (for exact BFMI):")
        print("   In your NumPyro MCMC run, add:")
        print("   mcmc = MCMC(kernel, num_warmup=..., num_samples=...,")
        print("              num_chains=..., chain_method='parallel')")
        print("   mcmc.run(..., return_info=True)  # This saves energy info")
        print("   # Or use: mcmc.run(..., extra_fields=['energy'])")
    
    return bfmi

def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_fit_metrics.py <state_file.npz> [--plots] [--divergences] [--bfmi]")
        print("Options:")
        print("  --plots       Create diagnostic plots")
        print("  --divergences Check for divergent transitions")
        print("  --bfmi        Compute BFMI (Bayesian Fraction of Missing Information)")
        return
    
    state_file = sys.argv[1]
    create_plots = '--plots' in sys.argv
    check_div = '--divergences' in sys.argv
    check_bfmi = '--bfmi' in sys.argv
    
    if not Path(state_file).exists():
        print(f"File not found: {state_file}")
        return
    
    # Basic assessment
    results = quick_assessment(state_file)
    
    # Check divergences
    if check_div:
        check_divergences(state_file)
    
    # Compute BFMI
    if check_bfmi:
        compute_bfmi(state_file)
    
    # Create plots
    if create_plots:
        create_diagnostic_plots(state_file)

if __name__ == "__main__":
    main() 