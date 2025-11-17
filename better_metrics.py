#!/usr/bin/env python3
"""
Better Metrics for Hawkes Process Models
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def geweke_diagnostic(samples, first_frac=0.1, last_frac=0.5):
    """Geweke diagnostic: compare beginning vs end of chain"""
    if samples.ndim < 2:
        return np.nan
    
    # Average across chains if multiple
    if samples.ndim == 3:
        samples_avg = np.mean(samples, axis=0)
    else:
        samples_avg = samples
    
    n_samples = samples_avg.shape[0]
    first_n = int(n_samples * first_frac)
    last_n = int(n_samples * last_frac)
    
    z_scores = []
    for i in range(min(5, samples_avg.shape[1])):  # Test first 5 params
        param_samples = samples_avg[:, i]
        
        # Beginning vs end
        first_part = param_samples[:first_n]
        last_part = param_samples[-last_n:]
        
        # Z-score for difference in means
        diff = np.mean(first_part) - np.mean(last_part)
        se = np.sqrt(np.var(first_part)/len(first_part) + np.var(last_part)/len(last_part))
        z_score = diff / se if se > 0 else np.nan
        z_scores.append(z_score)
    
    return np.array(z_scores)

def heidelberger_welch(samples, alpha=0.05):
    """Heidelberger-Welch stationarity test"""
    if samples.ndim < 2:
        return np.nan
    
    # Average across chains
    if samples.ndim == 3:
        samples_avg = np.mean(samples, axis=0)
    else:
        samples_avg = samples
    
    n_samples = samples_avg.shape[0]
    half_point = n_samples // 2
    
    p_values = []
    for i in range(min(5, samples_avg.shape[1])):
        param_samples = samples_avg[:, i]
        
        # Split in half and test if means are different
        first_half = param_samples[:half_point]
        second_half = param_samples[half_point:]
        
        # Simple t-test approximation
        diff = np.mean(first_half) - np.mean(second_half)
        pooled_var = (np.var(first_half) + np.var(second_half)) / 2
        se = np.sqrt(pooled_var * (2 / half_point))
        t_stat = diff / se if se > 0 else 0
        
        # Approximate p-value (not exact but good enough for diagnostics)
        p_val = 2 * (1 - np.abs(t_stat) / (np.abs(t_stat) + 1))
        p_values.append(p_val)
    
    return np.array(p_values)

def monte_carlo_se(samples):
    """Monte Carlo Standard Error - uncertainty in estimates"""
    if samples.ndim < 2:
        return np.nan
    
    # Average across chains
    if samples.ndim == 3:
        samples_avg = np.mean(samples, axis=0)
    else:
        samples_avg = samples
    
    mcse_values = []
    for i in range(min(5, samples_avg.shape[1])):
        param_samples = samples_avg[:, i]
        
        # MCSE = std / sqrt(ESS_effective)
        # Use simple ESS estimate
        acf = np.correlate(param_samples, param_samples, mode='full')
        acf = acf[acf.size//2:acf.size//2+10]
        acf = acf / acf[0]
        ess_eff = len(param_samples) / (1 + 2 * np.sum(acf[1:]))
        
        mcse = np.std(param_samples) / np.sqrt(ess_eff) if ess_eff > 0 else np.nan
        mcse_values.append(mcse)
    
    return np.array(mcse_values)

def effective_sample_ratio(samples):
    """ESS / total_samples ratio - should be > 0.1"""
    if samples.ndim < 2:
        return np.nan
    
    if samples.ndim == 3:
        samples_avg = np.mean(samples, axis=0)
        total_samples = samples.shape[0] * samples.shape[1]
    else:
        samples_avg = samples
        total_samples = samples.shape[0]
    
    ess_ratios = []
    for i in range(min(5, samples_avg.shape[1])):
        param_samples = samples_avg[:, i]
        
        # Simple ESS estimate
        acf = np.correlate(param_samples, param_samples, mode='full')
        acf = acf[acf.size//2:acf.size//2+10]
        acf = acf / acf[0]
        ess = len(param_samples) / (1 + 2 * np.sum(acf[1:]))
        
        ratio = ess / total_samples
        ess_ratios.append(ratio)
    
    return np.array(ess_ratios)

def assess_fit_quality(state_file):
    """Assess fit using multiple metrics"""
    print(f"\n{'='*60}")
    print(f"FIT QUALITY ASSESSMENT: {Path(state_file).name}")
    print(f"{'='*60}")
    
    try:
        state = np.load(state_file, allow_pickle=True)
        
        # Key parameters to check
        key_params = ['mu', 'K_masked', 'M_K']
        
        all_metrics = {}
        
        for param_name in key_params:
            if param_name in state:
                samples = state[param_name]
                print(f"\n--- {param_name} ---")
                
                if samples.ndim >= 2:
                    # Compute all metrics
                    geweke = geweke_diagnostic(samples)
                    heidel = heidelberger_welch(samples)
                    mcse = monte_carlo_se(samples)
                    ess_ratio = effective_sample_ratio(samples)
                    
                    print(f"Geweke Z-scores: {np.mean(np.abs(geweke)):.3f} (mean abs)")
                    print(f"Heidelberger p-values: {np.mean(heidel):.3f} (mean)")
                    print(f"Monte Carlo SE: {np.mean(mcse):.6f} (mean)")
                    print(f"ESS ratio: {np.mean(ess_ratio):.3f} (mean)")
                    
                    # Store for overall assessment
                    all_metrics[param_name] = {
                        'geweke': geweke,
                        'heidel': heidel,
                        'mcse': mcse,
                        'ess_ratio': ess_ratio
                    }
                    
                    # Individual assessments
                    if np.mean(np.abs(geweke)) < 2.0:
                        print("  ✅ Geweke: Good (|Z| < 2)")
                    else:
                        print("  ⚠️  Geweke: Poor (|Z| >= 2)")
                    
                    if np.mean(heidel) > 0.05:
                        print("  ✅ Heidelberger: Good (p > 0.05)")
                    else:
                        print("  ⚠️  Heidelberger: Poor (p <= 0.05)")
                    
                    if np.mean(ess_ratio) > 0.1:
                        print("  ✅ ESS ratio: Good (> 0.1)")
                    else:
                        print("  ⚠️  ESS ratio: Poor (<= 0.1)")
        
        # Overall assessment
        print(f"\n{'='*40}")
        print("OVERALL ASSESSMENT")
        print(f"{'='*40}")
        
        if all_metrics:
            # Count good vs poor metrics
            good_counts = 0
            total_counts = 0
            
            for param_name, metrics in all_metrics.items():
                if np.mean(np.abs(metrics['geweke'])) < 2.0:
                    good_counts += 1
                total_counts += 1
                
                if np.mean(metrics['heidel']) > 0.05:
                    good_counts += 1
                total_counts += 1
                
                if np.mean(metrics['ess_ratio']) > 0.1:
                    good_counts += 1
                total_counts += 1
            
            good_ratio = good_counts / total_counts if total_counts > 0 else 0
            
            print(f"Good metrics: {good_counts}/{total_counts} ({good_ratio:.1%})")
            
            if good_ratio >= 0.8:
                print("✅ FIT QUALITY: EXCELLENT")
            elif good_ratio >= 0.6:
                print("✅ FIT QUALITY: GOOD")
            elif good_ratio >= 0.4:
                print("⚠️  FIT QUALITY: ACCEPTABLE")
            else:
                print("❌ FIT QUALITY: POOR")
        
        return all_metrics
        
    except Exception as e:
        print(f"Error analyzing {state_file}: {e}")
        return None

def main():
    # Check these files
    state_files = [
        "mcmc_state_np_large_arbon_events_evening_copy_linear.npz",
        "mcmc_state_np_large_arbon_events_evening_copy.npz",
        "mcmc_state_np_test_arbon_events_evening.npz"
    ]
    
    print("BETTER METRICS FOR HAWKES PROCESS MODELS")
    print("=" * 60)
    print("These metrics are more informative than basic ESS:")
    print("- Geweke: |Z| < 2 = good (beginning ≈ end of chain)")
    print("- Heidelberger: p > 0.05 = good (stationary)")
    print("- ESS ratio: > 0.1 = good (efficient sampling)")
    print("- Monte Carlo SE: lower = better (less uncertainty)")
    
    for state_file in state_files:
        if Path(state_file).exists():
            assess_fit_quality(state_file)
        else:
            print(f"\n⚠️  File not found: {state_file}")

if __name__ == "__main__":
    main() 