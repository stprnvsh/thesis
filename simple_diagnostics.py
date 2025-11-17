#!/usr/bin/env python3
"""
Simple Bayesian Diagnostics for Hawkes Process Inference Results
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

def compute_rhat_simple(samples):
    """Simple R-hat computation for MCMC samples"""
    if samples.ndim < 2:
        return np.nan
    
    # For 3D arrays: (chains, samples, parameters)
    if samples.ndim == 3:
        n_chains, n_samples = samples.shape[:2]
        if n_chains == 1:
            return np.nan
        
        # Compute between-chain variance
        chain_means = np.mean(samples, axis=1)  # (chains, parameters)
        overall_mean = np.mean(chain_means, axis=0)  # (parameters)
        B = n_samples * np.var(chain_means, axis=0, ddof=1)  # (parameters)
        
        # Compute within-chain variance
        chain_vars = np.var(samples, axis=1, ddof=1)  # (chains, parameters)
        W = np.mean(chain_vars, axis=0)  # (parameters)
        
        # Compute R-hat
        var_plus = (n_samples - 1) / n_samples * W + 1 / n_samples * B
        rhat = np.sqrt(var_plus / W)
        
        return rhat
    
    return np.nan

def compute_ess_simple(samples):
    """Simple ESS computation using autocorrelation"""
    if samples.ndim < 2:
        return np.nan
    
    # For 3D arrays, compute ESS for each parameter separately
    if samples.ndim == 3:
        n_chains, n_samples, n_params = samples.shape
        
        # Average across chains first
        samples_avg = np.mean(samples, axis=0)  # (samples, parameters)
        
        ess_list = []
        for param_idx in range(n_params):
            param_samples = samples_avg[:, param_idx]
            
            # Compute autocorrelation
            acf = np.correlate(param_samples, param_samples, mode='full')
            acf = acf[acf.size//2:]  # Take positive lags only
            
            # Normalize
            acf = acf / acf[0]
            
            # Find first crossing of 0.05
            threshold = 0.05
            first_crossing = np.where(acf < threshold)[0]
            if len(first_crossing) > 0:
                lag = first_crossing[0]
            else:
                lag = min(len(acf) - 1, 50)  # Cap at 50 to avoid infinite loops
            
            # ESS = N / (1 + 2*sum(autocorrelations))
            ess = len(param_samples) / (1 + 2 * np.sum(acf[1:lag+1]))
            ess_list.append(ess)
        
        return np.array(ess_list)
    
    return np.nan

def analyze_mcmc_state(state_file):
    """Analyze MCMC state file and print diagnostics"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {state_file}")
    print(f"{'='*60}")
    
    try:
        state = np.load(state_file, allow_pickle=True)
        print(f"Available arrays: {list(state.keys())}")
        
        # Key parameters to analyze
        key_params = ['mu', 'K_masked', 'M_K', 'alpha', 'a_uncon', 'b_uncon']
        
        for param_name in key_params:
            if param_name in state:
                samples = state[param_name]
                print(f"\n--- {param_name} ---")
                print(f"Shape: {samples.shape}")
                
                if samples.ndim >= 2:
                    # Compute basic statistics
                    mean_val = np.mean(samples, axis=0)
                    std_val = np.std(samples, axis=0)
                    
                    print(f"Mean shape: {mean_val.shape}")
                    print(f"Std shape: {std_val.shape}")
                    
                    # Compute R-hat if multiple chains
                    rhat = compute_rhat_simple(samples)
                    if not np.isnan(rhat).all():
                        print(f"R-hat: {np.mean(rhat):.3f} (mean), {np.min(rhat):.3f} (min), {np.max(rhat):.3f} (max)")
                    
                    # Compute ESS
                    ess = compute_ess_simple(samples)
                    if not np.isnan(ess).all():
                        print(f"ESS: {np.mean(ess):.1f} (mean), {np.min(ess):.1f} (min), {np.max(ess):.1f} (max)")
                    
                    # Show some sample values
                    if samples.ndim == 3:  # (chains, samples, params)
                        print(f"Sample values (first chain, first sample): {samples[0, 0, :5] if samples.shape[2] > 5 else samples[0, 0, :]}")
                    elif samples.ndim == 2:  # (samples, params)
                        print(f"Sample values (first sample): {samples[0, :5] if samples.shape[1] > 5 else samples[0, :]}")
                
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
        
        return state
        
    except Exception as e:
        print(f"Error loading {state_file}: {e}")
        return None

def main():
    # Files to analyze based on what's available
    files_to_check = [
        "mcmc_state_np3_test_arbon_events_evening.npz",
        "mcmc_state_np_large_arbon_events_evening_copy_linear.npz", 
        "mcmc_state_np_large_arbon_events_evening_copy.npz",
        "mcmc_state_np_test_arbon_events_evening.npz"
    ]
    
    for state_file in files_to_check:
        if Path(state_file).exists():
            analyze_mcmc_state(state_file)
        else:
            print(f"\n⚠️  State file not found: {state_file}")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("To get proper Bayesian diagnostics, you need:")
    print("1. MCMC state files (.npz) - contain full posterior samples")
    print("2. Inference result files (.pickle) - contain posterior means only")
    print("\nKey diagnostics available:")
    print("- R-hat (Gelman-Rubin): < 1.1 indicates good convergence")
    print("- ESS (Effective Sample Size): Higher is better")
    print("- Trace plots: Visual convergence assessment")
    print("- Parameter summaries with credible intervals")

if __name__ == "__main__":
    main() 